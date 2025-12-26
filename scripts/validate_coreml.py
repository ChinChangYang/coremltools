#!/usr/bin/env python3
"""Cross-validate Core ML model against KataGo Eigen backend.

This script compares the outputs of the converted Core ML model against
the KataGo C++ Eigen backend to verify conversion correctness.
"""

import argparse
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


def load_test_input(json_path: Path) -> dict:
    """Load test input from JSON file."""
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
    """Run Core ML model inference."""
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
                # (1, 2, 19, 19) -> take channel 0 -> (19, 19)
                arr = arr[0, 0, :, :]
            elif mapped_key == "pass_policy":
                # (1, 2) -> take element [0, 0] -> (1,)
                arr = np.array([arr[0, 0]])
            elif mapped_key == "value":
                # (1, 3) -> (3,)
                arr = arr.squeeze()
            elif mapped_key == "ownership":
                # (1, 1, 19, 19) -> (19, 19)
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
    input_json = {
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


def print_comparison_results(test_name: str, results: dict) -> bool:
    """Print comparison results in a formatted way.

    Returns:
        True if all tests passed, False otherwise
    """
    print(f"\n=== Test Case: {test_name} ===")

    all_passed = True
    for key, result in results.items():
        status = result["status"]
        if status == "PASS":
            status_str = f"\033[92mPASS\033[0m"  # Green
        elif status == "MISSING":
            status_str = f"\033[93mMISSING\033[0m"  # Yellow
            all_passed = False
        else:
            status_str = f"\033[91mFAIL\033[0m"  # Red
            all_passed = False

        if status in ("PASS", "FAIL", "FAIL_NAN_INF"):
            print(
                f"  {key}: max_diff={result['max_diff']:.2e}, "
                f"mean_diff={result['mean_diff']:.2e}, "
                f"tol={result['tolerance']:.0e} [{status_str}]"
            )
            if result.get("has_nan"):
                print(f"    WARNING: Contains NaN values")
            if result.get("has_inf"):
                print(f"    WARNING: Contains Inf values")
        elif status == "SHAPE_MISMATCH":
            print(
                f"  {key}: shape mismatch - eigen={result['eigen_shape']}, "
                f"coreml={result['coreml_shape']} [{status_str}]"
            )
        else:
            print(f"  {key}: {result.get('message', status)} [{status_str}]")

    return all_passed


def run_coreml_only_test(model_path: str, inputs: dict, test_name: str) -> None:
    """Run Core ML model only and print outputs (when Eigen is not available)."""
    print(f"\n=== Test Case: {test_name} (Core ML only) ===")

    outputs = run_coreml_model(model_path, inputs)

    for key, value in outputs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Min: {value.min():.6f}, Max: {value.max():.6f}")
            print(f"    Mean: {value.mean():.6f}, Std: {value.std():.6f}")
            has_nan = np.isnan(value).any()
            has_inf = np.isinf(value).any()
            if has_nan:
                print(f"    WARNING: Contains NaN values")
            if has_inf:
                print(f"    WARNING: Contains Inf values")
        else:
            print(f"  {key}: {type(value)}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate Core ML model against KataGo Eigen backend"
    )
    parser.add_argument(
        "--model-mlpackage",
        type=str,
        required=True,
        help="Path to converted Core ML .mlpackage model",
    )
    parser.add_argument(
        "--model-bin",
        type=str,
        default=None,
        help="Path to KataGo .bin.gz model file (for Eigen comparison)",
    )
    parser.add_argument(
        "--katago-exe",
        type=str,
        default=None,
        help="Path to the KataGo executable (built with Eigen backend)",
    )
    parser.add_argument(
        "--test-inputs",
        type=str,
        default="test_inputs",
        help="Directory containing test input JSON files",
    )
    parser.add_argument(
        "--test-case",
        type=str,
        default=None,
        help="Run only a specific test case (by name)",
    )
    parser.add_argument(
        "--coreml-only",
        action="store_true",
        help="Run Core ML model only (skip Eigen comparison)",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Output results to JSON file",
    )
    args = parser.parse_args()

    # Validate arguments
    if not args.coreml_only and (args.model_bin is None or args.katago_exe is None):
        print("Warning: --model-bin and --katago-exe not specified, running Core ML only mode")
        args.coreml_only = True

    # Find test input files
    test_inputs_dir = Path(args.test_inputs)
    if not test_inputs_dir.exists():
        print(f"Error: Test inputs directory not found: {test_inputs_dir}")
        print("Run generate_test_inputs.py first to create test cases.")
        sys.exit(1)

    test_files = sorted(test_inputs_dir.glob("*.json"))
    test_files = [f for f in test_files if f.name != "index.json"]

    if args.test_case:
        test_files = [f for f in test_files if f.stem == args.test_case]
        if not test_files:
            print(f"Error: Test case '{args.test_case}' not found")
            sys.exit(1)

    if not test_files:
        print("Error: No test input files found")
        sys.exit(1)

    print(f"Found {len(test_files)} test case(s)")
    print(f"Core ML model: {args.model_mlpackage}")
    if not args.coreml_only:
        print(f"KataGo model: {args.model_bin}")
        print(f"KataGo exe: {args.katago_exe}")

    # Run tests
    all_results = {}
    all_passed = True

    for test_file in test_files:
        inputs = load_test_input(test_file)
        test_name = inputs["name"]

        if args.coreml_only:
            run_coreml_only_test(args.model_mlpackage, inputs, test_name)
        else:
            # Run both backends
            coreml_out = run_coreml_model(args.model_mlpackage, inputs)
            eigen_out = run_eigen_backend(args.model_bin, inputs, args.katago_exe)

            if eigen_out is None:
                print(f"\n=== Test Case: {test_name} ===")
                print("  Skipped: Eigen backend not available or failed")
                all_passed = False
                continue

            # Compare outputs
            results = compare_outputs(eigen_out, coreml_out)
            all_results[test_name] = results

            if not print_comparison_results(test_name, results):
                all_passed = False

    # Output JSON results if requested
    if args.json_output and all_results:
        with open(args.json_output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.json_output}")

    # Summary
    print("\n" + "=" * 50)
    if args.coreml_only:
        print("Core ML inference completed (no comparison performed)")
    elif all_passed:
        print("\033[92mAll tests PASSED\033[0m")
    else:
        print("\033[91mSome tests FAILED\033[0m")
        sys.exit(1)


if __name__ == "__main__":
    main()
