#!/usr/bin/env python3
"""
Test script for mask elimination optimization.

Converts KataGo model with eliminate_identity_mask enabled.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import coremltools as ct
from coremltools.converters import katago

def main():
    print("=" * 60)
    print("Testing Mask Elimination Optimization")
    print("=" * 60)
    print()

    # Model path used across all experiments
    model_path = "/Users/chinchangyang/katago_workspace/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"

    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        print("Please ensure the model file exists.")
        return 1

    print(f"Converting model: {model_path}")
    print(f"Configuration:")
    print(f"  - eliminate_identity_mask: True")
    print(f"  - minimum_deployment_target: iOS15")
    print(f"  - compute_precision: FLOAT16")
    print()

    try:
        # Convert with mask elimination
        mlmodel = katago.convert(
            model_path,
            minimum_deployment_target=ct.target.iOS15,
            compute_precision=ct.precision.FLOAT16,
            eliminate_identity_mask=True
        )

        output_path = "KataGo-mask_elimination.mlpackage"
        print(f"Saving model to: {output_path}")
        mlmodel.save(output_path)

        print()
        print("✓ Conversion successful!")
        print()
        print("Next steps:")
        print(f"  1. Validate: python scripts/validate_coreml.py --model-mlpackage {output_path}")
        print(f"  2. Benchmark: python scripts/benchmark_inference.py --model {output_path} --runs 100 --format json --output /tmp/mask_elimination_results.json")

        return 0

    except Exception as e:
        print(f"ERROR: Conversion failed!")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
