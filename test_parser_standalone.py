#!/usr/bin/env python3
"""Standalone test script to verify KataGo parser works without full coremltools."""

import sys
import gzip
import struct
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple, Optional

# Copy of the necessary type definitions
class ActivationType(IntEnum):
    IDENTITY = 0
    RELU = 1
    MISH = 2

@dataclass
class ConvLayerDesc:
    name: str
    conv_y_size: int
    conv_x_size: int
    in_channels: int
    out_channels: int
    dilation_y: int
    dilation_x: int
    weights: np.ndarray

@dataclass
class BatchNormLayerDesc:
    name: str
    num_channels: int
    epsilon: float
    has_scale: bool
    has_bias: bool
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    bias: np.ndarray
    merged_scale: np.ndarray
    merged_bias: np.ndarray

@dataclass
class ActivationLayerDesc:
    name: str
    activation_type: ActivationType

@dataclass
class MatMulLayerDesc:
    name: str
    in_channels: int
    out_channels: int
    weights: np.ndarray

@dataclass
class MatBiasLayerDesc:
    name: str
    num_channels: int
    weights: np.ndarray

ORDINARY_BLOCK_KIND = 0
GLOBAL_POOLING_BLOCK_KIND = 2
NESTED_BOTTLENECK_BLOCK_KIND = 3


class KataGoModelParser:
    SUPPORTED_VERSIONS = (15, 16)
    BOARD_SIZE = 19

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._buffer = None
        self._pos = 0
        self._binary_floats = True

    def parse(self):
        if self.model_path.endswith('.gz'):
            with gzip.open(self.model_path, 'rb') as f:
                self._buffer = f.read()
        else:
            with open(self.model_path, 'rb') as f:
                self._buffer = f.read()

        self._pos = 0
        self._binary_floats = b'@BIN@' in self._buffer
        return self._parse_model()

    def _read_until_whitespace(self) -> bytes:
        start = self._pos
        while self._pos < len(self._buffer):
            if self._buffer[self._pos:self._pos+1] in (b' ', b'\t', b'\n', b'\r'):
                break
            self._pos += 1
        return self._buffer[start:self._pos]

    def _skip_whitespace(self):
        while self._pos < len(self._buffer):
            if self._buffer[self._pos:self._pos+1] not in (b' ', b'\t', b'\n', b'\r'):
                break
            self._pos += 1

    def _read_string(self) -> str:
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return token.decode('utf-8')

    def _read_int(self) -> int:
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return int(token.decode('utf-8'))

    def _read_float(self) -> float:
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return float(token.decode('utf-8'))

    def _read_bool(self) -> bool:
        return self._read_int() != 0

    def _read_floats(self, count: int, name: str) -> np.ndarray:
        if not self._binary_floats:
            floats = np.empty(count, dtype=np.float32)
            for i in range(count):
                floats[i] = self._read_float()
            return floats
        else:
            while self._pos < len(self._buffer):
                if self._buffer[self._pos:self._pos+1] == b'@':
                    break
                self._pos += 1

            if self._buffer[self._pos:self._pos+5] != b'@BIN@':
                raise ValueError(f"{name}: expected @BIN@ marker for binary float block")
            self._pos += 5

            num_bytes = count * 4
            float_bytes = self._buffer[self._pos:self._pos + num_bytes]
            if len(float_bytes) != num_bytes:
                raise ValueError(f"{name}: expected {count} floats, got {len(float_bytes) // 4}")
            self._pos += num_bytes

            floats = np.frombuffer(float_bytes, dtype='<f4').copy()
            return floats.astype(np.float32)

    def _parse_conv_layer(self) -> ConvLayerDesc:
        name = self._read_string()
        conv_y_size = self._read_int()
        conv_x_size = self._read_int()
        in_channels = self._read_int()
        out_channels = self._read_int()
        dilation_y = self._read_int()
        dilation_x = self._read_int()

        num_weights = conv_y_size * conv_x_size * in_channels * out_channels
        weights_flat = self._read_floats(num_weights, name)
        weights = weights_flat.reshape(conv_y_size, conv_x_size, in_channels, out_channels)
        weights = np.transpose(weights, (3, 2, 0, 1)).copy()

        return ConvLayerDesc(
            name=name, conv_y_size=conv_y_size, conv_x_size=conv_x_size,
            in_channels=in_channels, out_channels=out_channels,
            dilation_y=dilation_y, dilation_x=dilation_x, weights=weights
        )

    def _parse_batchnorm_layer(self) -> BatchNormLayerDesc:
        name = self._read_string()
        num_channels = self._read_int()
        epsilon = self._read_float()
        has_scale = self._read_bool()
        has_bias = self._read_bool()

        mean = self._read_floats(num_channels, f"{name}/mean")
        variance = self._read_floats(num_channels, f"{name}/variance")

        if has_scale:
            scale = self._read_floats(num_channels, f"{name}/scale")
        else:
            scale = np.ones(num_channels, dtype=np.float32)

        if has_bias:
            bias = self._read_floats(num_channels, f"{name}/bias")
        else:
            bias = np.zeros(num_channels, dtype=np.float32)

        merged_scale = scale / np.sqrt(variance + epsilon)
        merged_bias = bias - merged_scale * mean

        return BatchNormLayerDesc(
            name=name, num_channels=num_channels, epsilon=epsilon,
            has_scale=has_scale, has_bias=has_bias, mean=mean,
            variance=variance, scale=scale, bias=bias,
            merged_scale=merged_scale, merged_bias=merged_bias
        )

    def _parse_activation_layer(self, model_version: int) -> ActivationLayerDesc:
        name = self._read_string()
        if model_version >= 11:
            activation_str = self._read_string()
            if activation_str == "ACTIVATION_IDENTITY":
                activation_type = ActivationType.IDENTITY
            elif activation_str == "ACTIVATION_RELU":
                activation_type = ActivationType.RELU
            elif activation_str == "ACTIVATION_MISH":
                activation_type = ActivationType.MISH
            else:
                raise ValueError(f"Unknown activation type: {activation_str}")
        else:
            activation_type = ActivationType.RELU
        return ActivationLayerDesc(name=name, activation_type=activation_type)

    def _parse_matmul_layer(self) -> MatMulLayerDesc:
        name = self._read_string()
        in_channels = self._read_int()
        out_channels = self._read_int()
        num_weights = in_channels * out_channels
        weights = self._read_floats(num_weights, name)
        weights = weights.reshape(in_channels, out_channels)
        return MatMulLayerDesc(name=name, in_channels=in_channels, out_channels=out_channels, weights=weights)

    def _parse_matbias_layer(self) -> MatBiasLayerDesc:
        name = self._read_string()
        num_channels = self._read_int()
        weights = self._read_floats(num_channels, name)
        return MatBiasLayerDesc(name=name, num_channels=num_channels, weights=weights)

    def _parse_residual_block(self, model_version: int):
        name = self._read_string()
        pre_bn = self._parse_batchnorm_layer()
        pre_act = self._parse_activation_layer(model_version)
        regular_conv = self._parse_conv_layer()
        mid_bn = self._parse_batchnorm_layer()
        mid_act = self._parse_activation_layer(model_version)
        final_conv = self._parse_conv_layer()
        return {"name": name, "pre_bn": pre_bn, "pre_act": pre_act, "regular_conv": regular_conv,
                "mid_bn": mid_bn, "mid_act": mid_act, "final_conv": final_conv}

    def _parse_global_pooling_residual_block(self, model_version: int):
        name = self._read_string()
        pre_bn = self._parse_batchnorm_layer()
        pre_act = self._parse_activation_layer(model_version)
        regular_conv = self._parse_conv_layer()
        gpool_conv = self._parse_conv_layer()
        gpool_bn = self._parse_batchnorm_layer()
        gpool_act = self._parse_activation_layer(model_version)
        gpool_to_bias_mul = self._parse_matmul_layer()
        mid_bn = self._parse_batchnorm_layer()
        mid_act = self._parse_activation_layer(model_version)
        final_conv = self._parse_conv_layer()
        return {"name": name, "pre_bn": pre_bn, "pre_act": pre_act, "regular_conv": regular_conv,
                "gpool_conv": gpool_conv, "gpool_bn": gpool_bn, "gpool_act": gpool_act,
                "gpool_to_bias_mul": gpool_to_bias_mul, "mid_bn": mid_bn, "mid_act": mid_act,
                "final_conv": final_conv}

    def _parse_block_stack(self, model_version: int, num_blocks: int, trunk_num_channels: int):
        blocks = []
        for _ in range(num_blocks):
            block_kind_name = self._read_string()
            if block_kind_name == "ordinary_block":
                block_kind = ORDINARY_BLOCK_KIND
                block = self._parse_residual_block(model_version)
            elif block_kind_name == "gpool_block":
                block_kind = GLOBAL_POOLING_BLOCK_KIND
                block = self._parse_global_pooling_residual_block(model_version)
            elif block_kind_name == "nested_bottleneck_block":
                block_kind = NESTED_BOTTLENECK_BLOCK_KIND
                block = self._parse_nested_bottleneck_block(model_version, trunk_num_channels)
            else:
                raise ValueError(f"Unknown block kind: {block_kind_name}")
            blocks.append((block_kind, block))
        return blocks

    def _parse_nested_bottleneck_block(self, model_version: int, trunk_num_channels: int):
        name = self._read_string()
        num_blocks = self._read_int()
        pre_bn = self._parse_batchnorm_layer()
        pre_act = self._parse_activation_layer(model_version)
        pre_conv = self._parse_conv_layer()
        blocks = self._parse_block_stack(model_version, num_blocks, pre_conv.out_channels)
        post_bn = self._parse_batchnorm_layer()
        post_act = self._parse_activation_layer(model_version)
        post_conv = self._parse_conv_layer()
        return {"name": name, "num_blocks": num_blocks, "pre_bn": pre_bn, "pre_act": pre_act,
                "pre_conv": pre_conv, "blocks": blocks, "post_bn": post_bn, "post_act": post_act,
                "post_conv": post_conv}

    def _parse_sgf_metadata_encoder(self, model_version: int, meta_encoder_version: int):
        name = self._read_string()
        mul1 = self._parse_matmul_layer()
        bias1 = self._parse_matbias_layer()
        act1 = self._parse_activation_layer(model_version)
        mul2 = self._parse_matmul_layer()
        bias2 = self._parse_matbias_layer()
        act2 = self._parse_activation_layer(model_version)
        mul3 = self._parse_matmul_layer()
        return {"name": name, "mul1": mul1, "bias1": bias1, "act1": act1,
                "mul2": mul2, "bias2": bias2, "act2": act2, "mul3": mul3}

    def _parse_trunk(self, model_version: int, meta_encoder_version: int):
        name = self._read_string()
        num_blocks = self._read_int()
        trunk_num_channels = self._read_int()
        mid_num_channels = self._read_int()
        regular_num_channels = self._read_int()
        _ = self._read_int()  # dilatedNumChannels (unused)
        gpool_num_channels = self._read_int()

        # Version >= 15 has 6 unused int parameters
        if model_version >= 15:
            for _ in range(6):
                self._read_int()

        initial_conv = self._parse_conv_layer()
        initial_matmul = self._parse_matmul_layer()

        sgf_metadata_encoder = None
        if meta_encoder_version > 0:
            sgf_metadata_encoder = self._parse_sgf_metadata_encoder(model_version, meta_encoder_version)

        blocks = self._parse_block_stack(model_version, num_blocks, trunk_num_channels)
        trunk_tip_bn = self._parse_batchnorm_layer()
        trunk_tip_act = self._parse_activation_layer(model_version)

        return {"name": name, "num_blocks": num_blocks, "trunk_num_channels": trunk_num_channels,
                "mid_num_channels": mid_num_channels, "regular_num_channels": regular_num_channels,
                "gpool_num_channels": gpool_num_channels, "initial_conv": initial_conv,
                "initial_matmul": initial_matmul, "sgf_metadata_encoder": sgf_metadata_encoder,
                "blocks": blocks, "trunk_tip_bn": trunk_tip_bn, "trunk_tip_act": trunk_tip_act}

    def _parse_policy_head(self, model_version: int):
        name = self._read_string()
        p1_conv = self._parse_conv_layer()
        g1_conv = self._parse_conv_layer()
        g1_bn = self._parse_batchnorm_layer()
        g1_act = self._parse_activation_layer(model_version)
        gpool_to_bias_mul = self._parse_matmul_layer()
        p1_bn = self._parse_batchnorm_layer()
        p1_act = self._parse_activation_layer(model_version)
        p2_conv = self._parse_conv_layer()
        gpool_to_pass_mul = self._parse_matmul_layer()

        gpool_to_pass_bias = None
        pass_act = None
        gpool_to_pass_mul2 = None
        if model_version >= 15:
            gpool_to_pass_bias = self._parse_matbias_layer()
            pass_act = self._parse_activation_layer(model_version)
            gpool_to_pass_mul2 = self._parse_matmul_layer()

        return {"name": name, "p1_conv": p1_conv, "g1_conv": g1_conv, "g1_bn": g1_bn,
                "g1_act": g1_act, "gpool_to_bias_mul": gpool_to_bias_mul, "p1_bn": p1_bn,
                "p1_act": p1_act, "p2_conv": p2_conv, "gpool_to_pass_mul": gpool_to_pass_mul,
                "gpool_to_pass_bias": gpool_to_pass_bias, "pass_act": pass_act,
                "gpool_to_pass_mul2": gpool_to_pass_mul2}

    def _parse_value_head(self, model_version: int):
        name = self._read_string()
        v1_conv = self._parse_conv_layer()
        v1_bn = self._parse_batchnorm_layer()
        v1_act = self._parse_activation_layer(model_version)
        v2_mul = self._parse_matmul_layer()
        v2_bias = self._parse_matbias_layer()
        v2_act = self._parse_activation_layer(model_version)
        v3_mul = self._parse_matmul_layer()
        v3_bias = self._parse_matbias_layer()
        sv3_mul = self._parse_matmul_layer()
        sv3_bias = self._parse_matbias_layer()
        v_ownership_conv = self._parse_conv_layer()

        return {"name": name, "v1_conv": v1_conv, "v1_bn": v1_bn, "v1_act": v1_act,
                "v2_mul": v2_mul, "v2_bias": v2_bias, "v2_act": v2_act,
                "v3_mul": v3_mul, "v3_bias": v3_bias, "sv3_mul": sv3_mul,
                "sv3_bias": sv3_bias, "v_ownership_conv": v_ownership_conv}

    def _parse_model(self):
        name = self._read_string()
        model_version = self._read_int()

        if model_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(f"Only KataGo model versions {self.SUPPORTED_VERSIONS} are supported, got {model_version}")

        num_input_channels = self._read_int()
        num_input_global_channels = self._read_int()

        # Parse post-process params (version >= 13)
        post_process_params = {}
        if model_version >= 13:
            post_process_params["td_score_multiplier"] = self._read_float()
            post_process_params["score_mean_multiplier"] = self._read_float()
            post_process_params["score_stdev_multiplier"] = self._read_float()
            post_process_params["lead_multiplier"] = self._read_float()
            post_process_params["variance_time_multiplier"] = self._read_float()
            post_process_params["shortterm_value_error_multiplier"] = self._read_float()
            post_process_params["shortterm_score_error_multiplier"] = self._read_float()

        # Parse meta encoder version (version >= 15)
        meta_encoder_version = 0
        if model_version >= 15:
            meta_encoder_version = self._read_int()
            for _ in range(7):
                self._read_int()

        trunk = self._parse_trunk(model_version, meta_encoder_version)
        policy_head = self._parse_policy_head(model_version)
        value_head = self._parse_value_head(model_version)

        return {
            "name": name,
            "model_version": model_version,
            "num_input_channels": num_input_channels,
            "num_input_global_channels": num_input_global_channels,
            "meta_encoder_version": meta_encoder_version,
            "post_process_params": post_process_params,
            "trunk": trunk,
            "policy_head": policy_head,
            "value_head": value_head
        }


def test_parser():
    """Test parsing the KataGo model."""
    model_path = "kata1-test.bin.gz"

    print(f"Parsing {model_path}...")
    parser = KataGoModelParser(model_path)

    try:
        model = parser.parse()
        print(f"Successfully parsed model: {model['name']}")
        print(f"  Model version: {model['model_version']}")
        print(f"  Input channels: {model['num_input_channels']}")
        print(f"  Global input channels: {model['num_input_global_channels']}")
        print(f"  Meta encoder version: {model['meta_encoder_version']}")

        trunk = model["trunk"]
        print(f"\nTrunk:")
        print(f"  Num blocks: {trunk['num_blocks']}")
        print(f"  Trunk channels: {trunk['trunk_num_channels']}")
        print(f"  Mid channels: {trunk['mid_num_channels']}")
        print(f"  Regular channels: {trunk['regular_num_channels']}")
        print(f"  GPool channels: {trunk['gpool_num_channels']}")
        print(f"  Initial conv shape: {trunk['initial_conv'].weights.shape}")
        print(f"  Initial matmul shape: {trunk['initial_matmul'].weights.shape}")

        ordinary_blocks = sum(1 for k, _ in trunk['blocks'] if k == 0)
        gpool_blocks = sum(1 for k, _ in trunk['blocks'] if k == 2)
        nested_blocks = sum(1 for k, _ in trunk['blocks'] if k == 3)
        print(f"  Ordinary blocks: {ordinary_blocks}")
        print(f"  Global pooling blocks: {gpool_blocks}")
        print(f"  Nested bottleneck blocks: {nested_blocks}")

        policy = model["policy_head"]
        print(f"\nPolicy head:")
        print(f"  P1 conv shape: {policy['p1_conv'].weights.shape}")
        print(f"  G1 conv shape: {policy['g1_conv'].weights.shape}")
        print(f"  P2 conv shape: {policy['p2_conv'].weights.shape}")

        value = model["value_head"]
        print(f"\nValue head:")
        print(f"  V1 conv shape: {value['v1_conv'].weights.shape}")
        print(f"  V2 mul shape: {value['v2_mul'].weights.shape}")
        print(f"  V3 mul shape: {value['v3_mul'].weights.shape}")
        print(f"  SV3 mul shape: {value['sv3_mul'].weights.shape}")
        print(f"  Ownership conv shape: {value['v_ownership_conv'].weights.shape}")

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
