#!/usr/bin/env python3
"""Benchmark KataGo Core ML model inference performance.

This script measures inference timing for optimization work by:
- Running warmup iterations to stabilize Neural Engine
- Collecting timing samples using high-resolution timer
- Computing robust statistics (median as primary metric)
- Supporting different compute units for comparison
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_test_input(test_case_path: Path) -> Dict:
    """Load test input from JSON file.

    Args:
        test_case_path: Path to JSON test case file

    Returns:
        Dictionary with 'name', 'description', 'spatial', 'global', 'mask' keys
    """
    with open(test_case_path) as f:
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
        "name": data.get("name", test_case_path.stem),
        "description": data.get("description", ""),
        "spatial": spatial,
        "global": global_in,
        "mask": mask,
    }


def compute_statistics(timings: List[float]) -> Dict[str, float]:
    """Compute statistical metrics from timing samples.

    Args:
        timings: List of timing measurements in milliseconds

    Returns:
        Dictionary with statistical metrics
    """
    return {
        "median_ms": float(np.median(timings)),
        "mean_ms": float(np.mean(timings)),
        "std_dev_ms": float(np.std(timings)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "p95_ms": float(np.percentile(timings, 95)),
    }


def benchmark_inference(
    model_path: str,
    inputs: Dict,
    num_runs: int,
    num_warmup: int,
    compute_unit: str,
) -> Dict:
    """Run inference benchmark and collect timing statistics.

    Args:
        model_path: Path to Core ML .mlpackage
        inputs: Test input dictionary from load_test_input()
        num_runs: Number of measurement runs
        num_warmup: Number of warmup runs (not timed)
        compute_unit: Compute unit string (CPU_ONLY, CPU_AND_NE, etc.)

    Returns:
        Dictionary with configuration and statistics
    """
    try:
        import coremltools as ct
    except ImportError:
        print(f"{Colors.YELLOW}Error: coremltools not installed. Please install it first.{Colors.END}")
        sys.exit(1)

    # Map compute unit string to enum
    compute_unit_map = {
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "ALL": ct.ComputeUnit.ALL,
    }

    cu = compute_unit_map.get(compute_unit)
    if cu is None:
        print(f"{Colors.YELLOW}Error: Invalid compute unit '{compute_unit}'{Colors.END}")
        sys.exit(1)

    # Load model
    print(f"Loading model... ", end="", flush=True)
    try:
        model = ct.models.MLModel(model_path, compute_units=cu)
        print(f"{Colors.GREEN}Done{Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}Failed{Colors.END}")
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Prepare Core ML inputs
    coreml_inputs = {
        "spatial_input": inputs["spatial"],
        "global_input": inputs["global"],
        "input_mask": inputs["mask"],
    }

    # Warmup runs
    if num_warmup > 0:
        print(f"Running {num_warmup} warmup iterations... ", end="", flush=True)
        try:
            for _ in range(num_warmup):
                _ = model.predict(coreml_inputs)
            print(f"{Colors.GREEN}Done{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}Failed{Colors.END}")
            print(f"Error during warmup: {e}")
            sys.exit(1)

    # Measurement runs
    print(f"Running {num_runs} measurement iterations... ", end="", flush=True)
    timings = []
    try:
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model.predict(coreml_inputs)
            end = time.perf_counter()
            timings.append((end - start) * 1000)  # Convert to milliseconds
        print(f"{Colors.GREEN}Done{Colors.END}")
    except Exception as e:
        print(f"{Colors.YELLOW}Failed{Colors.END}")
        print(f"Error during measurement: {e}")
        sys.exit(1)

    # Compute statistics
    stats = compute_statistics(timings)

    return {
        "config": {
            "model": model_path,
            "compute_unit": compute_unit,
            "test_case": inputs["name"],
            "test_description": inputs["description"],
            "num_runs": num_runs,
            "num_warmup": num_warmup,
        },
        "statistics": stats,
        "raw_timings_ms": timings,
    }


def print_text_results(results: Dict):
    """Print benchmark results in human-readable text format.

    Args:
        results: Dictionary from benchmark_inference()
    """
    config = results["config"]
    stats = results["statistics"]

    print()
    print(f"{Colors.BOLD}KataGo Core ML Inference Benchmark{Colors.END}")
    print("=" * 50)
    print()

    print(f"{Colors.BOLD}Configuration:{Colors.END}")
    print(f"  Model: {config['model']}")
    print(f"  Compute Unit: {config['compute_unit']}")
    print(f"  Test Case: {config['test_case']}")
    if config['test_description']:
        print(f"  Description: {config['test_description']}")
    print(f"  Warmup Runs: {config['num_warmup']}")
    print(f"  Measurement Runs: {config['num_runs']}")
    print()

    print(f"{Colors.BOLD}Results:{Colors.END}")
    print("-" * 50)
    print(f"  {Colors.CYAN}Median:    {stats['median_ms']:6.1f} ms{Colors.END}  {Colors.BOLD}← PRIMARY METRIC{Colors.END}")
    print(f"  Mean:      {stats['mean_ms']:6.1f} ms")
    print(f"  Std Dev:   {stats['std_dev_ms']:6.1f} ms")
    print(f"  Min:       {stats['min_ms']:6.1f} ms")
    print(f"  Max:       {stats['max_ms']:6.1f} ms")
    print(f"  P95:       {stats['p95_ms']:6.1f} ms")
    print()

    # Interpretation
    print(f"{Colors.BOLD}Interpretation:{Colors.END}")

    # Check variability
    cv = (stats['std_dev_ms'] / stats['median_ms']) * 100  # Coefficient of variation
    if cv < 5:
        print(f"  {Colors.GREEN}✓{Colors.END} Low variability ({cv:.1f}%) indicates consistent performance")
    elif cv < 10:
        print(f"  {Colors.YELLOW}○{Colors.END} Moderate variability ({cv:.1f}%) - acceptable for most use cases")
    else:
        print(f"  {Colors.YELLOW}!{Colors.END} High variability ({cv:.1f}%) - consider closing background apps")

    # P95 interpretation
    p95_overhead = ((stats['p95_ms'] - stats['median_ms']) / stats['median_ms']) * 100
    print(f"  {Colors.GREEN}✓{Colors.END} P95 ({stats['p95_ms']:.1f} ms) shows typical worst-case latency (+{p95_overhead:.1f}%)")

    # Min-max range
    range_ms = stats['max_ms'] - stats['min_ms']
    print(f"  {Colors.BLUE}→{Colors.END} Range: {range_ms:.1f} ms (min-max spread)")

    print()
    print(f"{Colors.BOLD}Recommendation for optimization work:{Colors.END}")
    threshold_ms = stats['median_ms'] * 0.05  # 5% of median
    print(f"  Compare {Colors.CYAN}{Colors.BOLD}MEDIAN{Colors.END} values across runs. A change is meaningful if:")
    print(f"  - Improvement > 5% ({threshold_ms:.1f} ms in this case)")
    print(f"  - Reproducible across multiple benchmark runs")
    print()


def print_json_results(results: Dict, output_file: Optional[str] = None):
    """Print benchmark results in JSON format.

    Args:
        results: Dictionary from benchmark_inference()
        output_file: Optional file path to write JSON to (None = stdout)
    """
    output = {
        "config": results["config"],
        "statistics": results["statistics"],
        "raw_timings_ms": results["raw_timings_ms"],
    }

    if output_file:
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"{Colors.GREEN}Results written to: {output_file}{Colors.END}")
    else:
        print(json.dumps(output, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KataGo Core ML model inference performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (50 runs, Neural Engine)
  python scripts/benchmark_inference.py --model KataGo.mlpackage

  # Custom number of runs
  python scripts/benchmark_inference.py --model KataGo.mlpackage --runs 100

  # Compare CPU vs Neural Engine
  python scripts/benchmark_inference.py --model KataGo.mlpackage --compute-unit CPU_ONLY
  python scripts/benchmark_inference.py --model KataGo.mlpackage --compute-unit CPU_AND_NE

  # Use specific test case
  python scripts/benchmark_inference.py --model KataGo.mlpackage --test-case random_seed_42

  # JSON output for CI/CD
  python scripts/benchmark_inference.py --model KataGo.mlpackage --format json --output results.json
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to Core ML .mlpackage model"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=50,
        help="Number of inference runs for measurement (default: 50)"
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup runs before measurement (default: 10)"
    )

    parser.add_argument(
        "--compute-unit",
        type=str,
        choices=["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"],
        default="CPU_AND_NE",
        help="Compute unit to use (default: CPU_AND_NE for Neural Engine)"
    )

    parser.add_argument(
        "--test-inputs",
        type=str,
        default="test_inputs",
        help="Directory containing test input JSON files (default: test_inputs)"
    )

    parser.add_argument(
        "--test-case",
        type=str,
        default="zeros",
        help="Test case to use (default: zeros - simplest realistic baseline)"
    )

    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for JSON format (default: print to stdout)"
    )

    args = parser.parse_args()

    # Validate model path
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"{Colors.YELLOW}Error: Model not found at '{args.model}'{Colors.END}")
        print("Please provide a valid path to a .mlpackage file")
        sys.exit(1)

    # Find test input file
    test_inputs_dir = Path(args.test_inputs)
    test_case_path = test_inputs_dir / f"{args.test_case}.json"

    if not test_case_path.exists():
        print(f"{Colors.YELLOW}Error: Test case '{args.test_case}' not found{Colors.END}")
        print(f"Expected path: {test_case_path}")
        print()
        print("Generate test inputs first:")
        print("  python scripts/generate_test_inputs.py")
        sys.exit(1)

    # Load test input
    try:
        inputs = load_test_input(test_case_path)
    except Exception as e:
        print(f"{Colors.YELLOW}Error loading test input: {e}{Colors.END}")
        sys.exit(1)

    # Run benchmark
    results = benchmark_inference(
        model_path=str(model_path),
        inputs=inputs,
        num_runs=args.runs,
        num_warmup=args.warmup,
        compute_unit=args.compute_unit,
    )

    # Output results
    if args.format == "json":
        print_json_results(results, args.output)
    else:
        print_text_results(results)


if __name__ == "__main__":
    main()
