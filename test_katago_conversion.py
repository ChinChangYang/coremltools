#!/usr/bin/env python3
"""Test script to verify KataGo to Core ML conversion.

Run this after building coremltools:
    make build
    source scripts/env_activate.sh --python=3.11
    python test_katago_conversion.py
"""

import sys
import numpy as np


def test_conversion():
    """Test converting the KataGo model to Core ML."""
    import coremltools as ct

    model_path = "kata1-test.bin.gz"

    print(f"Converting {model_path} to Core ML...")

    try:
        # Convert the model
        mlmodel = ct.converters.katago.convert(
            model_path,
            minimum_deployment_target=ct.target.iOS15
        )

        print("Conversion successful!")
        print(f"  Inputs: {[inp.name for inp in mlmodel.input_description]}")
        print(f"  Outputs: {[out.name for out in mlmodel.output_description]}")

        # Save the model
        output_path = "KataGo.mlpackage"
        mlmodel.save(output_path)
        print(f"Model saved to {output_path}")

        # Test inference with random input
        print("\nTesting inference...")
        spatial_input = np.random.randn(1, 22, 19, 19).astype(np.float32)
        global_input = np.random.randn(1, 19).astype(np.float32)
        input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)

        result = mlmodel.predict({
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask
        })

        print("Inference outputs:")
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"  {key}: {type(value)}")

        print("\nConversion test PASSED!")
        return True

    except Exception as e:
        print(f"Conversion test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_conversion()
    sys.exit(0 if success else 1)
