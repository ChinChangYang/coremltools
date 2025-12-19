#!/usr/bin/env python3
"""Simple test script to verify KataGo parser works."""

import sys
sys.path.insert(0, '/Users/chinchangyang/Code/KataGoCoremltools')

from coremltools.converters.katago._katago_parser import KataGoModelParser

def test_parser():
    """Test parsing the KataGo model."""
    model_path = "kata1-test.bin.gz"

    print(f"Parsing {model_path}...")
    parser = KataGoModelParser(model_path)

    try:
        model_desc = parser.parse()
        print(f"Successfully parsed model: {model_desc.name}")
        print(f"  Model version: {model_desc.model_version}")
        print(f"  Input channels: {model_desc.num_input_channels}")
        print(f"  Global input channels: {model_desc.num_input_global_channels}")
        print(f"  Policy channels: {model_desc.num_policy_channels}")
        print(f"  Value channels: {model_desc.num_value_channels}")
        print(f"  Score value channels: {model_desc.num_score_value_channels}")
        print(f"  Ownership channels: {model_desc.num_ownership_channels}")
        print(f"  Meta encoder version: {model_desc.meta_encoder_version}")

        print(f"\nTrunk:")
        print(f"  Num blocks: {model_desc.trunk.num_blocks}")
        print(f"  Trunk channels: {model_desc.trunk.trunk_num_channels}")
        print(f"  Mid channels: {model_desc.trunk.mid_num_channels}")
        print(f"  Regular channels: {model_desc.trunk.regular_num_channels}")
        print(f"  GPool channels: {model_desc.trunk.gpool_num_channels}")
        print(f"  Initial conv shape: {model_desc.trunk.initial_conv.weights.shape}")
        print(f"  Initial matmul shape: {model_desc.trunk.initial_matmul.weights.shape}")

        # Count block types
        ordinary_blocks = sum(1 for k, _ in model_desc.trunk.blocks if k == 0)
        gpool_blocks = sum(1 for k, _ in model_desc.trunk.blocks if k == 2)
        nested_blocks = sum(1 for k, _ in model_desc.trunk.blocks if k == 3)
        print(f"  Ordinary blocks: {ordinary_blocks}")
        print(f"  Global pooling blocks: {gpool_blocks}")
        print(f"  Nested bottleneck blocks: {nested_blocks}")

        print(f"\nPolicy head:")
        print(f"  P1 conv shape: {model_desc.policy_head.p1_conv.weights.shape}")
        print(f"  G1 conv shape: {model_desc.policy_head.g1_conv.weights.shape}")
        print(f"  P2 conv shape: {model_desc.policy_head.p2_conv.weights.shape}")

        print(f"\nValue head:")
        print(f"  V1 conv shape: {model_desc.value_head.v1_conv.weights.shape}")
        print(f"  V2 mul shape: {model_desc.value_head.v2_mul.weights.shape}")
        print(f"  V3 mul shape: {model_desc.value_head.v3_mul.weights.shape}")
        print(f"  SV3 mul shape: {model_desc.value_head.sv3_mul.weights.shape}")
        print(f"  Ownership conv shape: {model_desc.value_head.v_ownership_conv.weights.shape}")

        print("\nParser test PASSED!")
        return True

    except Exception as e:
        print(f"Parser test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parser()
    sys.exit(0 if success else 1)
