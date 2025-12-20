#!/usr/bin/env python3
"""Cross-validate Core ML model against KataGo Eigen backend.

This script verifies that the converted Core ML model produces identical
raw outputs as the KataGo C++ Eigen backend for the same inputs.

Usage:
    python validate_coreml_eigen.py \
        --eigen-binary /path/to/eigenvalidation \
        --model-bin kata1-test.bin.gz \
        --model-coreml KataGo_validation.mlpackage

Prerequisites:
    1. Build KataGo Eigen validation binary (see plan for details)
    2. Convert model with for_validation=True:
       ct.converters.katago.convert(model_path, for_validation=True)
"""

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np


def compute_mask_features(mask: np.ndarray) -> np.ndarray:
    """Compute the 3 mask-derived global features.

    Args:
        mask: Input mask of shape (1, 1, 19, 19).

    Returns:
        Array of shape (3,) containing:
        - mask_sum: Sum of valid positions
        - mask_sum_sqrt_s14_m01: (sqrt(mask_sum) - 14) * 0.1
        - mask_sum_sqrt_s14_m01_sq_s01: ((sqrt(mask_sum) - 14) * 0.1)^2 - 0.1
    """
    mask_sum = np.sum(mask)
    mask_sum_sqrt_s14_m01 = (np.sqrt(mask_sum) - 14.0) * 0.1
    mask_sum_sqrt_s14_m01_sq_s01 = mask_sum_sqrt_s14_m01 ** 2 - 0.1
    return np.array(
        [mask_sum, mask_sum_sqrt_s14_m01, mask_sum_sqrt_s14_m01_sq_s01],
        dtype=np.float32
    )


def generate_test_inputs(test_case: str, num_spatial_ch: int = 22, num_global_ch: int = 19):
    """Generate test inputs for a specific test case.

    Args:
        test_case: Name of the test case.
        num_spatial_ch: Number of spatial input channels.
        num_global_ch: Number of base global input channels (before mask features).

    Returns:
        Tuple of:
        - spatial: (1, num_spatial_ch, 19, 19) - spatial features
        - global_input: (1, num_global_ch + 3) - 19 base + 3 mask features
        - mask: (1, 1, 19, 19) - valid position mask
    """
    board_size = 19

    if test_case == "zeros":
        spatial = np.zeros((1, num_spatial_ch, board_size, board_size), dtype=np.float32)
        global_base = np.zeros((1, num_global_ch), dtype=np.float32)
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    elif test_case == "random_seed_42":
        np.random.seed(42)
        spatial = np.random.randn(1, num_spatial_ch, board_size, board_size).astype(np.float32) * 0.1
        global_base = np.random.randn(1, num_global_ch).astype(np.float32) * 0.1
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    elif test_case == "single_stone_center":
        spatial = np.zeros((1, num_spatial_ch, board_size, board_size), dtype=np.float32)
        spatial[0, 0, :, :] = 1.0  # On-board mask channel
        spatial[0, 1, 9, 9] = 1.0  # Player stone at center
        global_base = np.zeros((1, num_global_ch), dtype=np.float32)
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    elif test_case == "corner_pattern":
        spatial = np.zeros((1, num_spatial_ch, board_size, board_size), dtype=np.float32)
        spatial[0, 0, :, :] = 1.0  # On-board mask
        # Place stones at corners
        spatial[0, 1, 0, 0] = 1.0
        spatial[0, 1, 0, 18] = 1.0
        spatial[0, 1, 18, 0] = 1.0
        spatial[0, 1, 18, 18] = 1.0
        global_base = np.zeros((1, num_global_ch), dtype=np.float32)
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    elif test_case == "partial_mask":
        spatial = np.zeros((1, num_spatial_ch, board_size, board_size), dtype=np.float32)
        global_base = np.zeros((1, num_global_ch), dtype=np.float32)
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)
        # Zero out top half - simulating a partial board
        mask[0, 0, :9, :] = 0.0

    elif test_case == "all_ones_spatial":
        spatial = np.ones((1, num_spatial_ch, board_size, board_size), dtype=np.float32)
        global_base = np.zeros((1, num_global_ch), dtype=np.float32)
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    elif test_case == "large_values":
        np.random.seed(123)
        spatial = np.random.randn(1, num_spatial_ch, board_size, board_size).astype(np.float32) * 10.0
        global_base = np.random.randn(1, num_global_ch).astype(np.float32) * 10.0
        mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

    else:
        raise ValueError(f"Unknown test case: {test_case}")

    # Compute and append mask features to make extended global input
    mask_features = compute_mask_features(mask)
    global_input = np.concatenate([global_base, mask_features.reshape(1, 3)], axis=1)

    return spatial, global_input, mask


def save_inputs_for_eigen(
    spatial: np.ndarray,
    global_input: np.ndarray,
    mask: np.ndarray,
    output_path: str
):
    """Save inputs in binary format for Eigen validation binary.

    Format (little-endian):
    - int32: num_spatial_channels
    - int32: num_global_channels (including mask features)
    - int32: board_size
    - float32[]: spatial data (NCHW -> flattened)
    - float32[]: global data (flattened)
    - float32[]: mask data (NCHW -> flattened)

    Args:
        spatial: Spatial input of shape (1, C, H, W).
        global_input: Global input of shape (1, G).
        mask: Mask of shape (1, 1, H, W).
        output_path: Path to save binary file.
    """
    num_spatial_ch = spatial.shape[1]
    num_global_ch = global_input.shape[1]
    board_size = spatial.shape[2]

    with open(output_path, 'wb') as f:
        # Write header
        f.write(struct.pack('<i', num_spatial_ch))
        f.write(struct.pack('<i', num_global_ch))
        f.write(struct.pack('<i', board_size))

        # Write data (flattened, as float32)
        f.write(spatial.astype(np.float32).tobytes())
        f.write(global_input.astype(np.float32).tobytes())
        f.write(mask.astype(np.float32).tobytes())


def parse_eigen_output(stdout: str) -> Optional[Dict[str, np.ndarray]]:
    """Parse raw outputs from Eigen validation binary stdout.

    Expected format:
    RAW_OUTPUT_START
    POLICY: val1 val2 val3 ...
    PASS_POLICY: val1 val2 ...
    VALUE: val1 val2 val3
    OWNERSHIP: val1 val2 ...
    SCORE_VALUE: val1 val2 ...
    RAW_OUTPUT_END

    Args:
        stdout: Standard output from Eigen binary.

    Returns:
        Dictionary of output arrays, or None if parsing fails.
    """
    lines = stdout.strip().split('\n')

    outputs = {}
    in_output_section = False

    for line in lines:
        line = line.strip()

        if line == "RAW_OUTPUT_START":
            in_output_section = True
            continue
        elif line == "RAW_OUTPUT_END":
            in_output_section = False
            break

        if not in_output_section:
            continue

        if ':' in line:
            key, values_str = line.split(':', 1)
            key = key.strip().lower()
            values = [float(v) for v in values_str.strip().split()]
            outputs[key] = np.array(values, dtype=np.float32)

    if not outputs:
        return None

    return outputs


def run_coreml(
    model_path: str,
    spatial: np.ndarray,
    global_input: np.ndarray,
    mask: np.ndarray
) -> Dict[str, np.ndarray]:
    """Run Core ML inference and return raw outputs.

    Args:
        model_path: Path to .mlpackage file.
        spatial: Spatial input.
        global_input: Global input (with mask features).
        mask: Input mask.

    Returns:
        Dictionary of output arrays.
    """
    import coremltools as ct

    model = ct.models.MLModel(model_path)
    result = model.predict({
        "spatial_input": spatial,
        "global_input": global_input,
        "input_mask": mask
    })

    return {
        "policy": np.array(result["policy"]).flatten(),
        "pass_policy": np.array(result["pass_policy"]).flatten(),
        "value": np.array(result["value"]).flatten(),
        "ownership": np.array(result["ownership"]).flatten(),
        "score_value": np.array(result["score_value"]).flatten()
    }


def run_eigen(
    eigen_binary: str,
    model_path: str,
    spatial: np.ndarray,
    global_input: np.ndarray,
    mask: np.ndarray
) -> Optional[Dict[str, np.ndarray]]:
    """Run Eigen backend and parse raw output.

    Args:
        eigen_binary: Path to eigenvalidation binary.
        model_path: Path to .bin.gz model file.
        spatial: Spatial input.
        global_input: Global input (with mask features).
        mask: Input mask.

    Returns:
        Dictionary of output arrays, or None if failed.
    """
    # Save inputs to temp file
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        input_path = f.name

    try:
        save_inputs_for_eigen(spatial, global_input, mask, input_path)

        # Run Eigen validation binary
        result = subprocess.run(
            [eigen_binary, model_path, input_path],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"Eigen binary failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return None

        # Parse raw outputs from stdout
        outputs = parse_eigen_output(result.stdout)

        if outputs is None:
            print("Failed to parse Eigen output")
            print(f"stdout: {result.stdout}")
            return None

        return outputs

    finally:
        # Clean up temp file
        if os.path.exists(input_path):
            os.remove(input_path)


def compare_outputs(
    coreml_out: Dict[str, np.ndarray],
    eigen_out: Dict[str, np.ndarray],
    tolerances: Optional[Dict[str, float]] = None
) -> Tuple[Dict[str, dict], bool]:
    """Compare outputs with tolerance.

    Args:
        coreml_out: Core ML outputs.
        eigen_out: Eigen outputs.
        tolerances: Per-output tolerance values.

    Returns:
        Tuple of (results dict, all_passed bool).
    """
    if tolerances is None:
        tolerances = {
            "policy": 1e-4,
            "pass_policy": 1e-4,
            "value": 1e-5,
            "ownership": 1e-4,
            "score_value": 1e-4
        }

    results = {}
    all_passed = True

    for key in coreml_out:
        if key not in eigen_out:
            print(f"  {key}: MISSING in Eigen output")
            results[key] = {"passed": False, "error": "missing"}
            all_passed = False
            continue

        coreml_arr = coreml_out[key]
        eigen_arr = eigen_out[key]

        # Check shapes match
        if coreml_arr.shape != eigen_arr.shape:
            # Try to reshape if total elements match
            if coreml_arr.size == eigen_arr.size:
                eigen_arr = eigen_arr.reshape(coreml_arr.shape)
            else:
                print(f"  {key}: SHAPE MISMATCH - Core ML: {coreml_arr.shape}, Eigen: {eigen_arr.shape}")
                results[key] = {
                    "passed": False,
                    "error": f"shape mismatch: {coreml_arr.shape} vs {eigen_arr.shape}"
                }
                all_passed = False
                continue

        diff = np.abs(coreml_arr - eigen_arr)
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        tolerance = tolerances.get(key, 1e-4)
        passed = max_diff < tolerance

        results[key] = {
            "passed": passed,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "tolerance": tolerance
        }

        status = "PASS" if passed else "FAIL"
        print(f"  {key}: {status} (max_diff={max_diff:.2e}, mean_diff={mean_diff:.2e}, tolerance={tolerance:.0e})")

        if not passed:
            all_passed = False

            # Show some sample differences for debugging
            if len(diff.flatten()) > 10:
                worst_indices = np.argsort(diff.flatten())[-5:]
                print(f"    Worst differences at indices: {worst_indices}")
                for idx in worst_indices:
                    print(f"      [{idx}]: Core ML={coreml_arr.flatten()[idx]:.8f}, "
                          f"Eigen={eigen_arr.flatten()[idx]:.8f}, diff={diff.flatten()[idx]:.8f}")

    return results, all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validate Core ML model against KataGo Eigen backend"
    )
    parser.add_argument(
        "--eigen-binary",
        required=True,
        help="Path to eigenvalidation binary"
    )
    parser.add_argument(
        "--model-bin",
        required=True,
        help="Path to KataGo .bin.gz model file"
    )
    parser.add_argument(
        "--model-coreml",
        required=True,
        help="Path to Core ML .mlpackage file (converted with for_validation=True)"
    )
    parser.add_argument(
        "--test-cases",
        nargs="+",
        default=["zeros", "random_seed_42", "single_stone_center",
                 "corner_pattern", "partial_mask"],
        help="Test cases to run"
    )
    parser.add_argument(
        "--num-spatial-channels",
        type=int,
        default=22,
        help="Number of spatial input channels"
    )
    parser.add_argument(
        "--num-global-channels",
        type=int,
        default=19,
        help="Number of base global input channels (before mask features)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.eigen_binary):
        print(f"Error: Eigen binary not found: {args.eigen_binary}")
        return 1

    if not os.path.exists(args.model_bin):
        print(f"Error: Model .bin.gz not found: {args.model_bin}")
        return 1

    if not os.path.exists(args.model_coreml):
        print(f"Error: Core ML model not found: {args.model_coreml}")
        return 1

    print("=" * 60)
    print("KataGo Core ML vs Eigen Cross-Validation")
    print("=" * 60)
    print(f"Eigen binary: {args.eigen_binary}")
    print(f"Model (bin): {args.model_bin}")
    print(f"Model (Core ML): {args.model_coreml}")
    print(f"Test cases: {args.test_cases}")
    print()

    all_tests_passed = True

    for test_case in args.test_cases:
        print(f"\n{'=' * 40}")
        print(f"Test Case: {test_case}")
        print("=" * 40)

        # Generate test inputs
        spatial, global_input, mask = generate_test_inputs(
            test_case,
            num_spatial_ch=args.num_spatial_channels,
            num_global_ch=args.num_global_channels
        )

        print(f"Input shapes:")
        print(f"  spatial: {spatial.shape}")
        print(f"  global_input: {global_input.shape}")
        print(f"  mask: {mask.shape}")

        # Run Core ML inference
        print("\nRunning Core ML inference...")
        try:
            coreml_out = run_coreml(args.model_coreml, spatial, global_input, mask)
        except Exception as e:
            print(f"Core ML inference failed: {e}")
            all_tests_passed = False
            continue

        # Run Eigen inference
        print("Running Eigen inference...")
        eigen_out = run_eigen(args.eigen_binary, args.model_bin, spatial, global_input, mask)

        if eigen_out is None:
            print("Eigen inference failed")
            all_tests_passed = False
            continue

        # Compare outputs
        print("\nComparing outputs:")
        results, passed = compare_outputs(coreml_out, eigen_out)

        if not passed:
            all_tests_passed = False

    print(f"\n{'=' * 60}")
    if all_tests_passed:
        print("OVERALL: ALL TESTS PASSED")
    else:
        print("OVERALL: SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
