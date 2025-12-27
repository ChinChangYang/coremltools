#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Validation utilities for KataGo Core ML cross-validation tests.

This module provides reusable functions for comparing Core ML model outputs
against the KataGo Eigen backend, extracted from scripts/validate_coreml.py.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


# Tolerance thresholds for comparison
# Note: Float32 implementations can differ significantly between
# Core ML (ANE/GPU) and Eigen (CPU) due to operation ordering,
# fused operations, and numeric precision. These tolerances are
# set to allow for typical float32 accumulation differences.
TOLERANCES = {
    "policy": 7e-2,       # Policy logits can accumulate significant error
    "pass_policy": 1e-2,
    "value": 2e-1,        # Value head has many accumulated operations
    "ownership": 5e-3,
    "score_value": 5e-2,
}


def get_default_tolerances() -> dict:
    """Get default tolerance values for output comparison.

    Returns:
        Dictionary mapping output keys to tolerance thresholds
    """
    return TOLERANCES.copy()


def load_test_input(json_path: Path) -> dict:
    """Load test input from JSON file.

    Args:
        json_path: Path to JSON file containing test input

    Returns:
        Dictionary with keys: name, description, spatial, global, mask
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

    return {
        "name": data.get("name", json_path.stem),
        "description": data.get("description", ""),
        "spatial": spatial,
        "global": global_in,
        "mask": mask,
    }


def run_coreml_model(model_path: str, inputs: dict) -> dict:
    """Run Core ML model inference.

    Args:
        model_path: Path to .mlpackage Core ML model
        inputs: Dictionary with spatial, global, mask inputs

    Returns:
        Dictionary of output arrays with keys: policy, pass_policy, value, ownership, score_value
    """
    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools not installed. Please install it first.")
        sys.exit(1)

    model = ct.models.MLModel(model_path)

    result = model.predict({
        "spatial_input": inputs["spatial"],
        "global_input": inputs["global"],
        "input_mask": inputs["mask"],
    })

    # Map Core ML output names to standard names
    name_mapping = {
        "policy_p2_conv": "policy",
        "policy_pass_mul2": "pass_policy",
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
                # (1, 2, H, W) -> take channel 0 -> (H, W)
                arr = arr[0, 0, :, :]
            elif mapped_key == "pass_policy":
                # (1, 2) -> take element [0, 0] -> (1,)
                arr = np.array([arr[0, 0]])
            elif mapped_key == "value":
                # (1, 3) -> (3,)
                arr = arr.squeeze()
            elif mapped_key == "ownership":
                # (1, 1, H, W) -> (H, W)
                arr = arr.squeeze()
            elif mapped_key == "score_value":
                # (1, 6) -> (6,)
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
        tolerances: Dictionary of tolerance values per output key

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

        if eigen_flat.shape != coreml_flat.shape:
            results[key] = {
                "status": "SHAPE_MISMATCH",
                "eigen_shape": eigen_val.shape,
                "coreml_shape": coreml_val.shape,
            }
            continue

        diff = np.abs(eigen_flat - coreml_flat)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        tolerance = tolerances.get(key, 1e-4)

        # Check for NaN/Inf
        has_nan = bool(np.isnan(diff).any())
        has_inf = bool(np.isinf(diff).any())

        status = "PASS"
        if has_nan or has_inf:
            status = "FAIL_NAN_INF"
        elif max_diff >= tolerance:
            status = "FAIL"

        results[key] = {
            "status": status,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
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
