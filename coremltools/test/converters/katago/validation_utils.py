#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Validation utilities for KataGo Core ML cross-validation tests.

This module provides reusable functions for comparing Core ML model outputs
against the KataGo Eigen backend using relative tolerances for robust
cross-platform validation.
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import coremltools as ct


# Tolerance thresholds for comparison
# Tolerances are calculated as: max_diff / max(abs(eigen_reference))
# This normalizes the error relative to the magnitude of the reference output,
# making tolerances robust across different output ranges and scales.
# Values represent the maximum acceptable relative error as a fraction.
# Example: 1.3e-02 means maximum 1.3% relative error is acceptable.
TOLERANCES = {
    "policy": 1.3e-02,       # 1.3% relative error
    "pass_policy": 1.1e-02,  # 1.1% relative error
    "value": 8.8e-03,        # 0.88% relative error
    "ownership": 2.6e-02,    # 2.6% relative error
    "score_value": 1.1e-02,  # 1.1% relative error
}


def get_default_tolerances() -> dict:
    """Get default tolerance values for output comparison.

    Returns:
        Dictionary mapping output keys to relative tolerance thresholds
    """
    return TOLERANCES.copy()


def load_test_input(json_path: Path) -> dict:
    """Load test input from JSON file.

    Args:
        json_path: Path to JSON file containing test input

    Returns:
        Dictionary with keys: name, description, spatial, global, mask,
        and optionally 'meta' for human SL networks
    """
    with open(json_path) as f:
        data = json.load(f)

    # Convert lists to numpy arrays with proper shapes for Core ML
    spatial = np.array(data["spatial_input"], dtype=np.float32)
    global_in = np.array(data["global_input"], dtype=np.float32)
    mask = np.array(data["input_mask"], dtype=np.float32)

    # Add batch dimension
    spatial = spatial.reshape(1, *spatial.shape)  # [1, C, H, W]
    global_in = global_in.reshape(1, -1)  # [1, G]
    mask = mask.reshape(1, 1, *mask.shape)  # [1, 1, H, W]

    result = {
        "name": data.get("name", json_path.stem),
        "description": data.get("description", ""),
        "spatial": spatial,
        "global": global_in,
        "mask": mask,
    }

    # Load metadata input if present (for human SL networks)
    if "meta_input" in data:
        meta = np.array(data["meta_input"], dtype=np.float32)
        meta = meta.reshape(1, -1)  # [1, M]
        result["meta"] = meta

    return result


def run_coreml_model(
    model_path: str,
    inputs: dict,
    model: Optional["ct.models.MLModel"] = None,
) -> dict:
    """Run Core ML model inference.

    Args:
        model_path: Path to .mlpackage Core ML model
        inputs: Dictionary with spatial, global, mask inputs
        model: Optional pre-loaded Core ML model instance. If provided, this model
               will be used instead of loading from model_path. This improves performance
               when running inference on multiple inputs with the same model.

    Returns:
        Dictionary of output arrays with keys: policy, pass_policy, value, ownership, score_value
    """
    # Load model if not provided (supports both cached and fresh loading)
    if model is None:
        try:
            import coremltools as ct
        except ImportError:
            print("Error: coremltools not installed. Please install it first.")
            sys.exit(1)

        model = ct.models.MLModel(
            model_path,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
        )

    # Build prediction inputs
    predict_inputs = {
        "spatial_input": inputs["spatial"],
        "global_input": inputs["global"],
        "input_mask": inputs["mask"],
    }

    # Add metadata input if present (for human SL networks)
    if "meta" in inputs:
        predict_inputs["meta_input"] = inputs["meta"]
    else:
        # Check if model requires meta_input (human SL networks)
        # If so, provide default zero metadata
        model_spec = model.get_spec()
        input_names = [input.name for input in model_spec.description.input]
        if "meta_input" in input_names:
            # Human SL models use 192-channel metadata input
            predict_inputs["meta_input"] = np.zeros((1, 192), dtype=np.float32)

    result = model.predict(predict_inputs)

    # Map Core ML output names to standard names
    name_mapping = {
        "policy_p2_conv": "policy",
        "policy_pass_mul2": "pass_policy",  # v15+ (two-layer pass computation)
        "policy_pass": "pass_policy",        # v8-14 (single-layer pass computation)
        "value_v3_bias": "value",
        "value_ownership_conv": "ownership",
        "value_sv3_bias": "score_value",
    }

    # Convert to numpy arrays with mapped names and proper shapes
    outputs = {}
    for key, value in result.items():
        mapped_key = name_mapping.get(key, key)
        if hasattr(value, "__array__"):
            arr = np.array(value)
            # Reshape outputs to match Eigen backend format
            if mapped_key == "policy":
                # v8: (1, 1, H, W), v15+: (1, 2, H, W) or (1, 4, H, W)
                # Take channel 0 -> (H, W)
                arr = arr[0, 0, :, :]
            elif mapped_key == "pass_policy":
                # v8: (1, 1), v15+: (1, 2) or (1, 4)
                # Take element [0, 0] -> (1,)
                arr = np.array([arr[0, 0]])
            elif mapped_key == "value":
                # (1, 3) -> (3,)
                arr = arr.squeeze()
            elif mapped_key == "ownership":
                # (1, 1, H, W) -> (H, W)
                arr = arr.squeeze()
            elif mapped_key == "score_value":
                # v8: (1, 4), v9+: (1, 6) -> squeeze to (4,) or (6,)
                arr = arr.squeeze()
            outputs[mapped_key] = arr
        else:
            outputs[mapped_key] = value

    return outputs


def run_eigen_backend(
    model_path: str,
    inputs: dict,
    katago_exe: str,
) -> Optional[dict]:
    """Run KataGo Eigen backend and get raw outputs.

    Args:
        model_path: Path to KataGo .bin.gz model file
        inputs: Dictionary with spatial, global, mask inputs
        katago_exe: Path to the KataGo executable (with validation subcommand)

    Returns:
        Dictionary of output arrays, or None if execution failed
    """
    # Convert inputs to JSON format (remove batch dimension for JSON)
    # Extract board dimensions from input shapes
    spatial_shape = inputs["spatial"].shape  # [1, C, H, W]
    board_y_size, board_x_size = spatial_shape[2], spatial_shape[3]

    input_json = {
        "boardXSize": int(board_x_size),       # Explicit board width
        "boardYSize": int(board_y_size),       # Explicit board height
        "spatial_input": inputs["spatial"].squeeze(0).tolist(),  # [C, H, W]
        "global_input": inputs["global"].squeeze(0).tolist(),  # [G]
        "input_mask": inputs["mask"].squeeze(0).squeeze(0).tolist(),  # [H, W]
    }

    # Add metadata input if present (for human SL networks)
    if "meta" in inputs:
        input_json["meta_input"] = inputs["meta"].squeeze(0).tolist()  # [M]

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(input_json, f)
        input_path = f.name

    try:
        # Run KataGo validation subcommand
        result = subprocess.run(
            [katago_exe, "validation", "-model", model_path, "-input", input_path],
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Try to parse JSON even if return code is non-zero
        # (KataGo validation may return non-zero but still produce valid output)
        try:
            output_json = json.loads(result.stdout)
        except json.JSONDecodeError:
            if result.returncode != 0:
                print(f"Eigen backend failed (exit code {result.returncode}): {result.stderr}")
            return None

        return {
            "policy": np.array(output_json["policy"]),
            "pass_policy": np.array(output_json["pass_policy"]),
            "value": np.array(output_json["value"]),
            "ownership": np.array(output_json["ownership"]),
            "score_value": np.array(output_json["score_value"]),
        }

    except subprocess.TimeoutExpired:
        print("Eigen backend timed out")
        return None
    except json.JSONDecodeError as e:
        print(f"Failed to parse Eigen output: {e}")
        return None
    except FileNotFoundError:
        print(f"KataGo executable not found: {katago_exe}")
        return None
    finally:
        # Clean up temp file
        Path(input_path).unlink(missing_ok=True)


def compare_outputs(
    eigen_out: dict,
    coreml_out: dict,
    tolerances: Optional[dict] = None,
) -> dict:
    """Compare outputs and return comparison results.

    Args:
        eigen_out: Dictionary of Eigen backend outputs
        coreml_out: Dictionary of Core ML model outputs
        tolerances: Dictionary of relative tolerance values per output key (if None, uses default)

    Returns:
        Dictionary with comparison results for each output
    """
    if tolerances is None:
        tolerances = TOLERANCES

    results = {}

    for key in ["policy", "pass_policy", "value", "ownership", "score_value"]:
        if key not in eigen_out or key not in coreml_out:
            results[key] = {
                "status": "MISSING",
                "message": f"Output '{key}' missing from one of the backends",
            }
            continue

        eigen_val = eigen_out[key]
        coreml_val = coreml_out[key]

        # Check for None values
        if eigen_val is None or coreml_val is None:
            results[key] = {
                "status": "MISSING",
                "message": f"Output '{key}' is None in one of the backends",
                "eigen_is_none": eigen_val is None,
                "coreml_is_none": coreml_val is None,
            }
            continue

        # Flatten for comparison if shapes differ slightly
        eigen_flat = np.asarray(eigen_val).flatten()
        coreml_flat = np.asarray(coreml_val).flatten()

        # Additional check after asarray conversion (in case of object arrays)
        if eigen_flat.dtype == object or coreml_flat.dtype == object:
            results[key] = {
                "status": "MISSING",
                "message": f"Output '{key}' contains None or object dtype",
                "eigen_dtype": str(eigen_flat.dtype),
                "coreml_dtype": str(coreml_flat.dtype),
            }
            continue

        # Special handling for score_value: Eigen backend outputs 6 channels for all models
        # but v8 models only have 4 channels. Compare only overlapping channels.
        if key == "score_value" and eigen_flat.shape != coreml_flat.shape:
            # Take the minimum length for comparison
            min_len = min(len(eigen_flat), len(coreml_flat))
            eigen_flat = eigen_flat[:min_len]
            coreml_flat = coreml_flat[:min_len]

        if eigen_flat.shape != coreml_flat.shape:
            results[key] = {
                "status": "SHAPE_MISMATCH",
                "eigen_shape": eigen_val.shape,
                "coreml_shape": coreml_val.shape,
            }
            continue

        # Calculate absolute differences
        diff = np.abs(eigen_flat - coreml_flat)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))

        # Calculate relative differences (relative to Eigen reference)
        # Avoid division by zero by using a small epsilon
        eigen_abs_max = float(np.max(np.abs(eigen_flat)))
        epsilon = 1e-10
        if eigen_abs_max > epsilon:
            max_relative_diff = max_diff / eigen_abs_max
            mean_relative_diff = mean_diff / eigen_abs_max
        else:
            # If reference is near zero, relative error is undefined
            max_relative_diff = float('inf') if max_diff > epsilon else 0.0
            mean_relative_diff = float('inf') if mean_diff > epsilon else 0.0

        tolerance = tolerances.get(key, 1e-4)

        # Check for NaN/Inf
        has_nan = bool(np.isnan(diff).any())
        has_inf = bool(np.isinf(diff).any())

        # Determine status based on relative tolerance
        status = "PASS"
        if has_nan or has_inf:
            status = "FAIL_NAN_INF"
        elif max_relative_diff >= tolerance:
            status = "FAIL"

        results[key] = {
            "status": status,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "max_relative_diff": max_relative_diff,
            "mean_relative_diff": mean_relative_diff,
            "tolerance": tolerance,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "eigen_shape": list(eigen_val.shape),
            "coreml_shape": list(coreml_val.shape),
        }

    return results


def is_eigen_backend_available(katago_exe: Optional[str] = None) -> bool:
    """Check if KataGo Eigen backend is available and functional.

    Args:
        katago_exe: Path to KataGo executable (if None, returns False)

    Returns:
        True if Eigen backend is available and can run validation command
    """
    if katago_exe is None:
        return False

    try:
        # Try running katago with --help to check if it's executable
        result = subprocess.run(
            [katago_exe, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        # If we can run it and get help text, assume it's functional
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, PermissionError):
        return False


# ==============================================================================
# Eigen Backend Output Caching
# ==============================================================================


@lru_cache(maxsize=1)
def compute_model_hash(model_path: str) -> str:
    """Compute SHA-256 hash of model file (cached per session).

    Args:
        model_path: Path to KataGo .bin.gz model file

    Returns:
        First 16 characters of SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def compute_input_hash(inputs: dict) -> str:
    """Compute hash of input arrays for cache validation.

    Args:
        inputs: Dictionary with spatial, global, mask, and optionally meta inputs

    Returns:
        First 16 characters of SHA-256 hash of input data
    """
    sha256 = hashlib.sha256()
    for key in sorted(inputs.keys()):
        if key in ("spatial", "global", "mask", "meta"):
            sha256.update(inputs[key].tobytes())
    return sha256.hexdigest()[:16]


def get_eigen_cache_path(
    cache_dir: Path, model_hash: str, board_size: int, test_case: str
) -> Path:
    """Build cache file path for Eigen output.

    Args:
        cache_dir: Directory to store cache files
        model_hash: Hash of the model file
        board_size: Board size (e.g., 9, 13, 19)
        test_case: Test case name (e.g., "zeros", "random_seed_42")

    Returns:
        Path to cache file
    """
    return cache_dir / f"{model_hash}_{board_size}x{board_size}_{test_case}.json"


def load_cached_eigen_output(
    cache_path: Path, expected_input_hash: str
) -> Optional[dict]:
    """Load cached Eigen output if valid.

    Args:
        cache_path: Path to cache file
        expected_input_hash: Expected hash of input data for validation

    Returns:
        Dictionary of output arrays if cache is valid, None otherwise
    """
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            cache = json.load(f)
        if cache.get("input_hash") != expected_input_hash:
            return None  # Input changed, invalidate cache
        return {
            "policy": np.array(cache["outputs"]["policy"]),
            "pass_policy": np.array(cache["outputs"]["pass_policy"]),
            "value": np.array(cache["outputs"]["value"]),
            "ownership": np.array(cache["outputs"]["ownership"]),
            "score_value": np.array(cache["outputs"]["score_value"]),
        }
    except (json.JSONDecodeError, KeyError):
        return None  # Corrupted cache


def save_eigen_output_to_cache(
    cache_path: Path,
    model_hash: str,
    board_size: int,
    test_case: str,
    input_hash: str,
    outputs: dict,
) -> None:
    """Save Eigen output to cache.

    Args:
        cache_path: Path to cache file
        model_hash: Hash of the model file
        board_size: Board size (e.g., 9, 13, 19)
        test_case: Test case name
        input_hash: Hash of input data
        outputs: Dictionary of output arrays from Eigen backend
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        "model_hash": model_hash,
        "board_size": board_size,
        "test_case": test_case,
        "input_hash": input_hash,
        "outputs": {key: val.tolist() for key, val in outputs.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)


def run_eigen_backend_cached(
    model_path: str,
    inputs: dict,
    katago_exe: str,
    cache_dir: Path,
    test_case: str,
    board_size: int,
) -> Optional[dict]:
    """Run Eigen backend with caching support.

    This function first checks if a valid cached output exists for the given
    model, inputs, and test case. If so, it returns the cached output.
    Otherwise, it runs the Eigen backend and caches the result.

    Args:
        model_path: Path to KataGo .bin.gz model file
        inputs: Dictionary with spatial, global, mask inputs
        katago_exe: Path to the KataGo executable
        cache_dir: Directory to store cache files
        test_case: Test case name (e.g., "zeros", "random_seed_42")
        board_size: Board size (e.g., 9, 13, 19)

    Returns:
        Dictionary of output arrays, or None if execution failed
    """
    model_hash = compute_model_hash(model_path)
    input_hash = compute_input_hash(inputs)
    cache_path = get_eigen_cache_path(cache_dir, model_hash, board_size, test_case)

    # Try to load from cache
    cached = load_cached_eigen_output(cache_path, input_hash)
    if cached is not None:
        return cached

    # Cache miss - run Eigen backend
    outputs = run_eigen_backend(model_path, inputs, katago_exe)
    if outputs is not None:
        save_eigen_output_to_cache(
            cache_path, model_hash, board_size, test_case, input_hash, outputs
        )

    return outputs
