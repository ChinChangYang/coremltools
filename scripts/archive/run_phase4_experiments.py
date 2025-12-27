#!/usr/bin/env python3
"""
Run Phase 4 optimization experiments for KataGo Core ML converter.

Tests custom pass pipelines, skipping irrelevant passes, and const_elimination threshold.
"""

import json
import subprocess
import sys
import time

import coremltools as ct

# Model paths
KATAGO_MODEL_PATH = '/Users/chinchangyang/katago_workspace/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz'
RESULTS_FILE = 'scripts/optimization_results.json'


def create_skip_irrelevant_passes_pipeline():
    """
    Experiment 8: Create pipeline that skips passes irrelevant to KataGo.

    KataGo doesn't use GELU, PReLU, or LayerNorm, so we can skip those fusion passes.
    """
    pipeline = ct.PassPipeline.DEFAULT

    # Remove passes that are irrelevant to KataGo architecture
    irrelevant_passes = {
        "common::fuse_gelu_tanh_approximation",  # KataGo uses Mish, not GELU
        "common::fuse_gelu_exact",               # KataGo uses Mish, not GELU
        "common::fuse_prelu",                    # KataGo uses Mish, not PReLU
        "common::prelu_to_lrelu",                # KataGo uses Mish, not PReLU
        "common::fuse_layernorm_or_instancenorm",  # KataGo uses BatchNorm, not LayerNorm
    }

    pipeline.remove_passes(irrelevant_passes)
    return pipeline


def create_aggressive_const_elimination_pipeline():
    """
    Experiment 9: Adjust const_elimination threshold to be more aggressive.

    The default skip_const_by_size might be conservative. Try a higher threshold
    to allow more const elimination.
    """
    pipeline = ct.PassPipeline.DEFAULT

    # Set more aggressive const elimination threshold (default is typically around 1MB)
    # Try 10MB to allow eliminating larger constants
    pipeline.set_options(
        pass_name="common::const_elimination",
        options={"skip_const_by_size": "10485760"},  # 10MB in bytes
    )

    return pipeline


def create_optimized_fusion_order_pipeline():
    """
    Experiment 7: Custom pass pipeline with optimized fusion order.

    Try moving conv/batchnorm fusion earlier and running it more times.
    """
    pipeline = ct.PassPipeline.DEFAULT

    # Find the first occurrence of fuse_conv_batchnorm
    passes = pipeline.passes
    first_fuse_conv_bn_idx = passes.index("common::fuse_conv_batchnorm")

    # Insert an extra early fusion pass
    pipeline.insert_pass(first_fuse_conv_bn_idx, "common::fuse_conv_batchnorm")

    return pipeline


def convert_model_with_config(experiment_id, config):
    """Convert model with given configuration."""
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_id}")
    print(f"Config: {config}")
    print(f"{'='*80}\n")

    # Extract config parameters
    minimum_deployment_target = getattr(ct.target, config.get("minimum_deployment_target", "iOS15"))
    compute_precision = getattr(ct.precision, config.get("compute_precision", "FLOAT16")) if config.get("compute_precision") else None
    mish_implementation = config.get("mish_implementation", "original")
    use_fused_linear = config.get("use_fused_linear", False)

    # Create pass pipeline if specified
    pass_pipeline = None
    pipeline_type = config.get("pass_pipeline")
    if pipeline_type == "skip_irrelevant":
        pass_pipeline = create_skip_irrelevant_passes_pipeline()
        print(f"Using custom pipeline: Skip irrelevant passes")
        print(f"Removed passes: {5} irrelevant passes")
    elif pipeline_type == "aggressive_const_elimination":
        pass_pipeline = create_aggressive_const_elimination_pipeline()
        print(f"Using custom pipeline: Aggressive const elimination (10MB threshold)")
    elif pipeline_type == "optimized_fusion_order":
        pass_pipeline = create_optimized_fusion_order_pipeline()
        print(f"Using custom pipeline: Optimized fusion order")

    # Convert model
    print("Converting model...")
    start_time = time.time()
    mlmodel = ct.converters.katago.convert(
        KATAGO_MODEL_PATH,
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=compute_precision,
        mish_implementation=mish_implementation,
        use_fused_linear=use_fused_linear,
        pass_pipeline=pass_pipeline,
    )
    conversion_time = time.time() - start_time
    print(f"✓ Conversion completed in {conversion_time:.1f}s")

    # Save model
    output_name = f"KataGo-{experiment_id}.mlpackage"
    mlmodel.save(output_name)
    print(f"✓ Saved to {output_name}")

    return output_name


def validate_model(model_path):
    """Run validation script."""
    print(f"\nValidating {model_path}...")
    result = subprocess.run(
        ["python", "scripts/validate_coreml.py", "--model", model_path],
        capture_output=True,
        text=True
    )

    # Parse validation result
    if "All outputs passed validation" in result.stdout:
        validation_status = "PASS - all 9 test cases passed"
        print(f"✓ {validation_status}")
    elif "Some outputs failed validation" in result.stdout:
        # Extract which output failed
        lines = result.stdout.split('\n')
        for line in lines:
            if "exceeded threshold" in line or "FAILED" in line:
                validation_status = f"FAIL - {line.strip()}"
                print(f"✗ {validation_status}")
                break
        else:
            validation_status = "FAIL - validation failed"
            print(f"✗ {validation_status}")
    else:
        validation_status = "ERROR - could not parse validation output"
        print(f"✗ {validation_status}")

    return validation_status


def benchmark_model(model_path):
    """Run benchmark script."""
    print(f"\nBenchmarking {model_path}...")
    result = subprocess.run(
        ["python", "scripts/benchmark_inference.py", "--model", model_path, "--runs", "100"],
        capture_output=True,
        text=True
    )

    # Parse benchmark results from JSON output
    try:
        # The script outputs JSON, extract it
        lines = result.stdout.split('\n')
        json_start = None
        for i, line in enumerate(lines):
            if line.strip().startswith('{'):
                json_start = i
                break

        if json_start is not None:
            json_str = '\n'.join(lines[json_start:])
            data = json.loads(json_str)
            stats = data['statistics']

            results = {
                "median_ms": round(stats['median_ms'], 3),
                "mean_ms": round(stats['mean_ms'], 3),
                "std_dev_ms": round(stats['std_dev_ms'], 3),
                "min_ms": round(stats['min_ms'], 3),
                "max_ms": round(stats['max_ms'], 3),
                "p95_ms": round(stats['p95_ms'], 3),
            }

            print(f"✓ Median: {results['median_ms']:.3f}ms")
            return results
    except Exception as e:
        print(f"✗ Error parsing benchmark results: {e}")
        return None


def update_results_file(experiment_id, name, config, results, validation, notes, accuracy_risk="LOW"):
    """Update optimization_results.json with new experiment."""
    with open(RESULTS_FILE, 'r') as f:
        data = json.load(f)

    # Create experiment entry
    experiment = {
        "id": experiment_id,
        "name": name,
        "config": config,
        "results": results,
        "validation": validation,
        "accuracy_risk": accuracy_risk,
        "notes": notes
    }

    # Add to experiments list
    data['experiments'].append(experiment)

    # Save updated results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Updated {RESULTS_FILE}")


def run_experiment(experiment_id, name, config, notes, accuracy_risk="LOW"):
    """Run a complete experiment: convert, validate, benchmark, and record results."""
    print(f"\n{'#'*80}")
    print(f"# EXPERIMENT: {name}")
    print(f"{'#'*80}")

    try:
        # Convert
        model_path = convert_model_with_config(experiment_id, config)

        # Validate
        validation = validate_model(model_path)

        # Benchmark
        results = benchmark_model(model_path)

        if results:
            # Calculate improvement vs baseline
            baseline_median = 7.189  # From Phase 1
            improvement = ((baseline_median - results['median_ms']) / baseline_median) * 100

            if improvement > 0:
                notes += f" - {improvement:.1%} faster than baseline"
            else:
                notes += f" - {-improvement:.1%} slower than baseline"

        # Update results file
        update_results_file(experiment_id, name, config, results, validation, notes, accuracy_risk)

        return True

    except Exception as e:
        print(f"\n✗ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Phase 4 experiments."""
    print("="*80)
    print("PHASE 4: Low Priority Optimization Experiments")
    print("="*80)

    experiments = [
        {
            "id": "skip_irrelevant_passes",
            "name": "Skip irrelevant passes (GELU, PReLU, LayerNorm)",
            "config": {
                "mish_implementation": "original",
                "minimum_deployment_target": "iOS15",
                "compute_precision": "FLOAT16",
                "pass_pipeline": "skip_irrelevant"
            },
            "notes": "Remove 5 irrelevant fusion passes",
            "accuracy_risk": "LOW"
        },
        {
            "id": "aggressive_const_elim",
            "name": "Aggressive const elimination (10MB threshold)",
            "config": {
                "mish_implementation": "original",
                "minimum_deployment_target": "iOS15",
                "compute_precision": "FLOAT16",
                "pass_pipeline": "aggressive_const_elimination"
            },
            "notes": "Increase const_elimination threshold to 10MB",
            "accuracy_risk": "LOW"
        },
        {
            "id": "optimized_fusion_order",
            "name": "Optimized fusion pass order",
            "config": {
                "mish_implementation": "original",
                "minimum_deployment_target": "iOS15",
                "compute_precision": "FLOAT16",
                "pass_pipeline": "optimized_fusion_order"
            },
            "notes": "Add extra early conv/batchnorm fusion pass",
            "accuracy_risk": "LOW"
        },
    ]

    success_count = 0
    for exp in experiments:
        if run_experiment(
            exp["id"],
            exp["name"],
            exp["config"],
            exp["notes"],
            exp.get("accuracy_risk", "LOW")
        ):
            success_count += 1

    print(f"\n{'='*80}")
    print(f"Phase 4 Complete: {success_count}/{len(experiments)} experiments successful")
    print(f"{'='*80}\n")

    return 0 if success_count == len(experiments) else 1


if __name__ == "__main__":
    sys.exit(main())
