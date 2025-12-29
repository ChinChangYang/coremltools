#!/usr/bin/env python3
"""Analyze cross-validation tolerances for KataGo Core ML converter.

This script runs cross-validation tests and collects detailed statistics
about the differences between Core ML and Eigen backend outputs.

Usage:
    # Analyze all board sizes
    python scripts/analyze_tolerances.py

    # Analyze specific board size
    python scripts/analyze_tolerances.py --board-size 19

    # Save detailed results to JSON
    python scripts/analyze_tolerances.py --output tolerances.json

    # Show only summary
    python scripts/analyze_tolerances.py --summary-only
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

# Import validation utilities
from coremltools.test.converters.katago.validation_utils import (
    compare_outputs,
    get_default_tolerances,
    load_test_input,
    run_coreml_model,
    run_eigen_backend,
)


def is_full_board_mask(inputs: dict) -> bool:
    """Check if input mask represents a full board (all 1.0 values).

    The eliminate_identity_mask optimization precomputes constants assuming
    all mask values are 1.0. Test cases with partial masks (0.0 values) are
    incompatible with this optimization and should be skipped.

    Args:
        inputs: Dictionary with 'mask' key containing the input mask array

    Returns:
        True if all mask values are 1.0 (full board), False otherwise
    """
    mask = inputs["mask"]  # Shape: [1, 1, H, W]
    return bool(np.all(mask == 1.0))


def collect_test_statistics(
    board_size: int,
    model_path: str,
    katago_bin: str,
    katago_exe: str,
    test_inputs_dir: Path,
    eliminate_identity_mask: bool,
) -> List[Dict]:
    """Run all tests for a board size and collect statistics.

    Args:
        board_size: Board size (9, 13, or 19)
        model_path: Path to Core ML .mlpackage
        katago_bin: Path to KataGo .bin.gz model
        katago_exe: Path to KataGo executable
        test_inputs_dir: Directory containing test input JSON files
        eliminate_identity_mask: Value of eliminate_identity_mask used for this model

    Returns:
        List of dictionaries with test results (one per test case)
    """
    test_files = sorted(test_inputs_dir.glob("*.json"))
    test_files = [f for f in test_files if f.name != "index.json"]

    results = []

    # Load Core ML model once for all test cases (performance optimization)
    # This eliminates redundant disk I/O and initialization overhead
    try:
        import coremltools as ct
    except ImportError:
        print("Error: coremltools not installed. Please install it first.")
        import sys
        sys.exit(1)

    print(f"  Loading Core ML model: {Path(model_path).name}")
    model = ct.models.MLModel(
        model_path,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    print(f"  Model loaded successfully")

    for test_file in test_files:
        print(f"  Running: {test_file.stem}")

        inputs = load_test_input(test_file)

        # Skip partial mask tests when eliminate_identity_mask=True
        # (the optimization assumes full board with all mask values = 1.0)
        if eliminate_identity_mask and not is_full_board_mask(inputs):
            print(f"    Skipped (partial mask incompatible with eliminate_identity_mask)")
            continue

        try:
            # Pass pre-loaded model for performance
            coreml_out = run_coreml_model(model_path, inputs, model=model)
            eigen_out = run_eigen_backend(katago_bin, inputs, katago_exe)

            if eigen_out is None:
                print(f"    Skipped (Eigen backend failed)")
                continue

            comparison = compare_outputs(eigen_out, coreml_out)

            results.append({
                "test_case": test_file.stem,
                "board_size": board_size,
                "eliminate_identity_mask": eliminate_identity_mask,
                "comparison": comparison,
            })

        except Exception as e:
            print(f"    Error: {e}")
            continue

    return results


def aggregate_statistics(all_results: List[Dict]) -> Dict:
    """Aggregate statistics across all test cases.

    Args:
        all_results: List of test result dictionaries

    Returns:
        Dictionary with statistics per output type
    """
    by_output = defaultdict(lambda: {
        "max_diffs": [],
        "mean_diffs": [],
        "max_relative_diffs": [],
        "mean_relative_diffs": [],
    })

    for result in all_results:
        comparison = result["comparison"]
        for output_key, metrics in comparison.items():
            if metrics["status"] in ("PASS", "FAIL", "FAIL_NAN_INF"):
                by_output[output_key]["max_diffs"].append(metrics["max_diff"])
                by_output[output_key]["mean_diffs"].append(metrics["mean_diff"])
                by_output[output_key]["max_relative_diffs"].append(metrics["max_relative_diff"])
                by_output[output_key]["mean_relative_diffs"].append(metrics["mean_relative_diff"])

    stats = {}
    for output_key, values in by_output.items():
        max_diffs = np.array(values["max_diffs"])
        mean_diffs = np.array(values["mean_diffs"])
        max_relative_diffs = np.array(values["max_relative_diffs"])
        mean_relative_diffs = np.array(values["mean_relative_diffs"])

        stats[output_key] = {
            "num_tests": len(max_diffs),
            "max_diff": {
                "min": float(np.min(max_diffs)),
                "max": float(np.max(max_diffs)),
                "mean": float(np.mean(max_diffs)),
                "median": float(np.median(max_diffs)),
                "p95": float(np.percentile(max_diffs, 95)),
                "p99": float(np.percentile(max_diffs, 99)),
            },
            "mean_diff": {
                "min": float(np.min(mean_diffs)),
                "max": float(np.max(mean_diffs)),
                "mean": float(np.mean(mean_diffs)),
                "median": float(np.median(mean_diffs)),
                "p95": float(np.percentile(mean_diffs, 95)),
                "p99": float(np.percentile(mean_diffs, 99)),
            },
            "max_relative_diff": {
                "min": float(np.min(max_relative_diffs)),
                "max": float(np.max(max_relative_diffs)),
                "mean": float(np.mean(max_relative_diffs)),
                "median": float(np.median(max_relative_diffs)),
                "p95": float(np.percentile(max_relative_diffs, 95)),
                "p99": float(np.percentile(max_relative_diffs, 99)),
            },
            "mean_relative_diff": {
                "min": float(np.min(mean_relative_diffs)),
                "max": float(np.max(mean_relative_diffs)),
                "mean": float(np.mean(mean_relative_diffs)),
                "median": float(np.median(mean_relative_diffs)),
                "p95": float(np.percentile(mean_relative_diffs, 95)),
                "p99": float(np.percentile(mean_relative_diffs, 99)),
            },
        }

    return stats


def suggest_tolerances(stats: Dict, safety_margin: float = 1.5) -> Dict:
    """Suggest minimum tolerances with safety margin.

    Args:
        stats: Statistics dictionary from aggregate_statistics()
        safety_margin: Multiplier for P99 max_relative_diff (default: 1.5)

    Returns:
        Dictionary of suggested relative tolerances per output type
    """
    suggested = {}
    for output_key, values in stats.items():
        baseline = values["max_relative_diff"]["p99"]
        suggested[output_key] = baseline * safety_margin
    return suggested


def print_report(
    stats: Dict,
    suggested_tolerances: Dict,
    current_tolerances: Dict,
    summary_only: bool = False,
):
    """Print detailed analysis report.

    Args:
        stats: Statistics dictionary
        suggested_tolerances: Suggested tolerance values
        current_tolerances: Current tolerance values
        summary_only: If True, only print summary table
    """

    print("\n" + "=" * 80)
    print("KataGo Cross-Validation Tolerance Analysis")
    print("=" * 80)

    # Summary table
    print("\nSummary (across all test cases):\n")
    print(f"{'Output':<15} {'Tests':<8} {'Max Diff':<12} {'Cur Tol':<12} {'Sug Tol':<12} {'Status'}")
    print("-" * 75)

    for output_key in ["policy", "pass_policy", "value", "ownership", "score_value"]:
        if output_key not in stats:
            continue

        num_tests = stats[output_key]["num_tests"]
        max_observed = stats[output_key]["max_relative_diff"]["max"]
        current_tol = current_tolerances[output_key]
        suggested_tol = suggested_tolerances[output_key]

        if max_observed > current_tol:
            status = "FAIL"
        elif suggested_tol < current_tol:
            status = "TIGHT"
        else:
            status = "OK"

        print(f"{output_key:<15} {num_tests:<8} {max_observed:<12.2e} {current_tol:<12.2e} {suggested_tol:<12.2e} {status}")

    if summary_only:
        return

    # Detailed statistics
    print("\n" + "=" * 80)
    print("Detailed Statistics")
    print("=" * 80)

    for output_key in ["policy", "pass_policy", "value", "ownership", "score_value"]:
        if output_key not in stats:
            continue

        values = stats[output_key]
        print(f"\n{output_key}:")
        print(f"  Tests: {values['num_tests']}")
        print(f"  Absolute Max Diff:")
        print(f"    Min:    {values['max_diff']['min']:.2e}")
        print(f"    Mean:   {values['max_diff']['mean']:.2e}")
        print(f"    Median: {values['max_diff']['median']:.2e}")
        print(f"    P95:    {values['max_diff']['p95']:.2e}")
        print(f"    P99:    {values['max_diff']['p99']:.2e}")
        print(f"    Max:    {values['max_diff']['max']:.2e}")
        print(f"  Absolute Mean Diff:")
        print(f"    Min:    {values['mean_diff']['min']:.2e}")
        print(f"    Mean:   {values['mean_diff']['mean']:.2e}")
        print(f"    Median: {values['mean_diff']['median']:.2e}")
        print(f"    P95:    {values['mean_diff']['p95']:.2e}")
        print(f"    P99:    {values['mean_diff']['p99']:.2e}")
        print(f"    Max:    {values['mean_diff']['max']:.2e}")
        print(f"  Relative Max Diff:")
        print(f"    Min:    {values['max_relative_diff']['min']:.2e}")
        print(f"    Mean:   {values['max_relative_diff']['mean']:.2e}")
        print(f"    Median: {values['max_relative_diff']['median']:.2e}")
        print(f"    P95:    {values['max_relative_diff']['p95']:.2e}")
        print(f"    P99:    {values['max_relative_diff']['p99']:.2e}")
        print(f"    Max:    {values['max_relative_diff']['max']:.2e}")
        print(f"  Relative Mean Diff:")
        print(f"    Min:    {values['mean_relative_diff']['min']:.2e}")
        print(f"    Mean:   {values['mean_relative_diff']['mean']:.2e}")
        print(f"    Median: {values['mean_relative_diff']['median']:.2e}")
        print(f"    P95:    {values['mean_relative_diff']['p95']:.2e}")
        print(f"    P99:    {values['mean_relative_diff']['p99']:.2e}")
        print(f"    Max:    {values['mean_relative_diff']['max']:.2e}")

    # Recommendations
    print("\n" + "=" * 80)
    print("Recommendations")
    print("=" * 80)

    print("\nSuggested tolerance updates (validation_utils.py):")
    print("```python")
    print("TOLERANCES = {")
    for output_key in ["policy", "pass_policy", "value", "ownership", "score_value"]:
        if output_key in suggested_tolerances:
            suggested = suggested_tolerances[output_key]
            print(f"    \"{output_key}\": {suggested:.1e},  # {suggested*100:.2f}% relative error")
    print("}")
    print("```")

    print("\nInterpretation:")
    print("  - Suggested tolerances use P99 max_relative_diff * 1.5 safety margin")
    print("  - FAIL: Current tolerance exceeded (tests would fail)")
    print("  - TIGHT: Suggested tolerance lower than current (can tighten)")
    print("  - OK: Current tolerance appropriate")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-validation tolerances for KataGo Core ML converter"
    )
    parser.add_argument(
        "--board-size",
        type=int,
        choices=[9, 13, 19],
        default=None,
        help="Board size to analyze (default: all sizes)"
    )
    parser.add_argument(
        "--model-bin",
        type=str,
        default=None,
        help="Path to KataGo .bin.gz model file (default: auto-detect)"
    )
    parser.add_argument(
        "--katago-exe",
        type=str,
        default=None,
        help="Path to KataGo executable (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save detailed results to JSON file"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Show only summary table"
    )
    parser.add_argument(
        "--safety-margin",
        type=float,
        default=1.5,
        help="Safety margin multiplier for suggested tolerances (default: 1.5)"
    )
    parser.add_argument(
        "--eliminate-identity-mask",
        type=str,
        choices=["true", "false", "both"],
        default="both",
        help="Test with eliminate_identity_mask=True, False, or both (default: both)"
    )
    args = parser.parse_args()

    # Auto-detect paths
    repo_root = Path(__file__).parent.parent

    if args.model_bin is None:
        # Try multiple possible locations
        possible_paths = [
            repo_root / "KataGo" / "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
            repo_root.parent / "KataGo" / "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
            repo_root / "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
        ]
        for path in possible_paths:
            if path.exists():
                args.model_bin = str(path)
                break

    if args.katago_exe is None:
        # Try multiple possible locations
        possible_paths = [
            repo_root / "KataGo" / "cpp" / "build" / "katago",
            repo_root.parent / "KataGo" / "cpp" / "build" / "katago",
        ]
        for path in possible_paths:
            if path.exists():
                args.katago_exe = str(path)
                break

    # Validate paths
    if args.model_bin is None or not Path(args.model_bin).exists():
        print(f"Error: KataGo model not found")
        print("Please specify --model-bin or place model at one of these locations:")
        print("  - KataGo/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz")
        print("  - ../KataGo/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz")
        sys.exit(1)

    if args.katago_exe is None or not Path(args.katago_exe).exists():
        print(f"Error: KataGo executable not found")
        print("Please specify --katago-exe or build KataGo at one of these locations:")
        print("  - KataGo/cpp/build/katago")
        print("  - ../KataGo/cpp/build/katago")
        sys.exit(1)

    print(f"Using KataGo model: {args.model_bin}")
    print(f"Using KataGo executable: {args.katago_exe}")

    # Determine board sizes
    if args.board_size:
        board_sizes = [args.board_size]
    else:
        board_sizes = [9, 13, 19]

    # Determine eliminate_identity_mask settings to test
    if args.eliminate_identity_mask == "both":
        mask_settings = [True, False]
    else:
        mask_settings = [args.eliminate_identity_mask == "true"]

    # Convert models and collect statistics
    all_results = []

    import coremltools as ct
    from coremltools.converters.katago import convert

    for board_size in board_sizes:
        for eliminate_identity_mask in mask_settings:
            mask_str = "mask_true" if eliminate_identity_mask else "mask_false"
            print(f"\nAnalyzing {board_size}x{board_size} board with eliminate_identity_mask={eliminate_identity_mask}...")

            model_path = str(repo_root / f"KataGo_{board_size}x{board_size}_{mask_str}.mlpackage")

            if not Path(model_path).exists():
                print(f"  Converting model to {model_path}...")
                mlmodel = convert(
                    args.model_bin,
                    board_x_size=board_size,
                    board_y_size=board_size,
                    eliminate_identity_mask=eliminate_identity_mask,
                    minimum_deployment_target=ct.target.iOS18,
                    compute_precision=ct.precision.FLOAT16,
                    compute_units=ct.ComputeUnit.CPU_AND_NE,
                )
                mlmodel.save(model_path)
            else:
                print(f"  Using existing model: {model_path}")

            test_inputs_dir = repo_root / "test_inputs" / f"{board_size}x{board_size}"

            if not test_inputs_dir.exists():
                print(f"  Skipping (test inputs not found: {test_inputs_dir})")
                continue

            results = collect_test_statistics(
                board_size,
                model_path,
                args.model_bin,
                args.katago_exe,
                test_inputs_dir,
                eliminate_identity_mask,
            )

            all_results.extend(results)
            print(f"  Collected {len(results)} test results")

    if not all_results:
        print("\nError: No test results collected")
        sys.exit(1)

    # Aggregate and report
    print(f"\nAggregating statistics from {len(all_results)} tests...")
    stats = aggregate_statistics(all_results)

    # Get current tolerances
    current_tolerances = get_default_tolerances()

    # Generate suggestions
    suggested_tolerances = suggest_tolerances(stats, safety_margin=args.safety_margin)

    print_report(
        stats,
        suggested_tolerances,
        current_tolerances,
        args.summary_only,
    )

    # Save JSON if requested
    if args.output:
        output_data = {
            "num_tests": len(all_results),
            "board_sizes": list(set(r["board_size"] for r in all_results)),
            "eliminate_identity_mask_settings": list(set(r["eliminate_identity_mask"] for r in all_results)),
            "statistics": stats,
            "current_tolerances": current_tolerances,
            "suggested_tolerances": suggested_tolerances,
            "safety_margin": args.safety_margin,
            "raw_results": all_results,
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\nDetailed results saved to: {args.output}")

    # Exit status
    failures = []
    for output_key, values in stats.items():
        max_observed = values["max_relative_diff"]["max"]
        current_tol = current_tolerances[output_key]
        if max_observed > current_tol:
            failures.append(output_key)

    if failures:
        print(f"\nWARNING: Current tolerances exceeded for: {', '.join(failures)}")
        print("Consider updating tolerances in validation_utils.py")
        sys.exit(1)
    else:
        print("\nAll tests within current tolerances")


if __name__ == "__main__":
    main()
