# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Parser for KataGo model binary files.

Supports KataGo model versions 8-16 in binary format (.bin, .bin.gz).
"""

import gzip
import struct
from io import BytesIO
from typing import BinaryIO, List, Tuple

import numpy as np

from ._katago_types import (
    GLOBAL_POOLING_BLOCK_KIND,
    NESTED_BOTTLENECK_BLOCK_KIND,
    ORDINARY_BLOCK_KIND,
    ActivationLayerDesc,
    ActivationType,
    BatchNormLayerDesc,
    ConvLayerDesc,
    GlobalPoolingResidualBlockDesc,
    KataGoModelDesc,
    MatBiasLayerDesc,
    MatMulLayerDesc,
    ModelPostProcessParams,
    NestedBottleneckResidualBlockDesc,
    PolicyHeadDesc,
    ResidualBlockDesc,
    SGFMetadataEncoderDesc,
    TrunkDesc,
    ValueHeadDesc,
)


class KataGoModelParser:
    """
    Parser for KataGo neural network model files.

    Supports versions 8-16 models in binary format (.bin, .bin.gz).
    """

    SUPPORTED_VERSIONS = (8, 9, 10, 11, 12, 13, 14, 15, 16)

    def __init__(self, model_path: str):
        """
        Initialize the parser.

        Args:
            model_path: Path to the KataGo model file.
        """
        self.model_path = model_path
        self._buffer = None
        self._pos = 0
        self._binary_floats = True

    def parse(self) -> KataGoModelDesc:
        """
        Parse the model file and return a structured model description.

        Returns:
            KataGoModelDesc containing all model parameters.

        Raises:
            ValueError: If the model version is not supported.
            IOError: If the file cannot be read.
        """
        # Read file content
        if self.model_path.endswith('.gz'):
            with gzip.open(self.model_path, 'rb') as f:
                self._buffer = f.read()
        else:
            with open(self.model_path, 'rb') as f:
                self._buffer = f.read()

        self._pos = 0

        # Detect if binary format (check for @BIN@ marker)
        self._binary_floats = b'@BIN@' in self._buffer

        # Parse model
        return self._parse_model()

    def _read_until_whitespace(self) -> bytes:
        """Read bytes until whitespace is encountered."""
        start = self._pos
        while self._pos < len(self._buffer):
            if self._buffer[self._pos:self._pos+1] in (b' ', b'\t', b'\n', b'\r'):
                break
            self._pos += 1
        return self._buffer[start:self._pos]

    def _skip_whitespace(self):
        """Skip whitespace characters."""
        while self._pos < len(self._buffer):
            if self._buffer[self._pos:self._pos+1] not in (b' ', b'\t', b'\n', b'\r'):
                break
            self._pos += 1

    def _read_string(self) -> str:
        """Read a whitespace-delimited string."""
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return token.decode('utf-8')

    def _read_int(self) -> int:
        """Read an integer."""
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return int(token.decode('utf-8'))

    def _read_float(self) -> float:
        """Read a float."""
        self._skip_whitespace()
        token = self._read_until_whitespace()
        return float(token.decode('utf-8'))

    def _read_bool(self) -> bool:
        """Read a boolean (0 or 1)."""
        return self._read_int() != 0

    def _read_floats(self, count: int, name: str) -> np.ndarray:
        """
        Read an array of floats.

        For binary format, looks for @BIN@ marker and reads raw floats.
        For text format, reads space-separated float values.
        """
        if not self._binary_floats:
            # Text format
            floats = np.empty(count, dtype=np.float32)
            for i in range(count):
                floats[i] = self._read_float()
            return floats
        else:
            # Binary format - find @BIN@ marker
            # Skip whitespace before @
            while self._pos < len(self._buffer):
                if self._buffer[self._pos:self._pos+1] == b'@':
                    break
                self._pos += 1

            # Check for @BIN@ header
            if self._buffer[self._pos:self._pos+5] != b'@BIN@':
                raise ValueError(f"{name}: expected @BIN@ marker for binary float block")
            self._pos += 5

            # Read binary floats (little-endian)
            num_bytes = count * 4
            float_bytes = self._buffer[self._pos:self._pos + num_bytes]
            if len(float_bytes) != num_bytes:
                raise ValueError(f"{name}: expected {count} floats, got {len(float_bytes) // 4}")
            self._pos += num_bytes

            floats = np.frombuffer(float_bytes, dtype='<f4').copy()  # Little-endian float32
            return floats.astype(np.float32)

    def _parse_conv_layer(self) -> ConvLayerDesc:
        """Parse a convolutional layer."""
        name = self._read_string()
        conv_y_size = self._read_int()
        conv_x_size = self._read_int()
        in_channels = self._read_int()
        out_channels = self._read_int()
        dilation_y = self._read_int()
        dilation_x = self._read_int()

        # Read weights in file order: [y, x, ic, oc]
        num_weights = conv_y_size * conv_x_size * in_channels * out_channels
        weights_flat = self._read_floats(num_weights, name)

        # Reshape to [y, x, ic, oc]
        weights = weights_flat.reshape(conv_y_size, conv_x_size, in_channels, out_channels)

        # Transpose to Core ML / CUDA order: [oc, ic, y, x]
        weights = np.transpose(weights, (3, 2, 0, 1)).copy()

        return ConvLayerDesc(
            name=name,
            conv_y_size=conv_y_size,
            conv_x_size=conv_x_size,
            in_channels=in_channels,
            out_channels=out_channels,
            dilation_y=dilation_y,
            dilation_x=dilation_x,
            weights=weights
        )

    def _parse_batchnorm_layer(self) -> BatchNormLayerDesc:
        """Parse a batch normalization layer."""
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

        # Compute merged scale and bias
        merged_scale = scale / np.sqrt(variance + epsilon)
        merged_bias = bias - merged_scale * mean

        return BatchNormLayerDesc(
            name=name,
            num_channels=num_channels,
            epsilon=epsilon,
            has_scale=has_scale,
            has_bias=has_bias,
            mean=mean,
            variance=variance,
            scale=scale,
            bias=bias,
            merged_scale=merged_scale,
            merged_bias=merged_bias
        )

    def _parse_activation_layer(self, model_version: int) -> ActivationLayerDesc:
        """Parse an activation layer."""
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
            # Pre-v11 models only have ReLU
            activation_type = ActivationType.RELU

        return ActivationLayerDesc(name=name, activation_type=activation_type)

    def _parse_matmul_layer(self) -> MatMulLayerDesc:
        """Parse a matrix multiplication layer."""
        name = self._read_string()
        in_channels = self._read_int()
        out_channels = self._read_int()

        # Weights in [ic, oc] order
        num_weights = in_channels * out_channels
        weights = self._read_floats(num_weights, name)
        weights = weights.reshape(in_channels, out_channels)

        return MatMulLayerDesc(
            name=name,
            in_channels=in_channels,
            out_channels=out_channels,
            weights=weights
        )

    def _parse_matbias_layer(self) -> MatBiasLayerDesc:
        """Parse a bias layer."""
        name = self._read_string()
        num_channels = self._read_int()
        weights = self._read_floats(num_channels, name)

        return MatBiasLayerDesc(
            name=name,
            num_channels=num_channels,
            weights=weights
        )

    def _parse_residual_block(self, model_version: int) -> ResidualBlockDesc:
        """Parse a standard residual block."""
        name = self._read_string()
        pre_bn = self._parse_batchnorm_layer()
        pre_activation = self._parse_activation_layer(model_version)
        regular_conv = self._parse_conv_layer()
        mid_bn = self._parse_batchnorm_layer()
        mid_activation = self._parse_activation_layer(model_version)
        final_conv = self._parse_conv_layer()

        return ResidualBlockDesc(
            name=name,
            pre_bn=pre_bn,
            pre_activation=pre_activation,
            regular_conv=regular_conv,
            mid_bn=mid_bn,
            mid_activation=mid_activation,
            final_conv=final_conv
        )

    def _parse_global_pooling_residual_block(self, model_version: int) -> GlobalPoolingResidualBlockDesc:
        """Parse a global pooling residual block."""
        name = self._read_string()
        pre_bn = self._parse_batchnorm_layer()
        pre_activation = self._parse_activation_layer(model_version)
        regular_conv = self._parse_conv_layer()
        gpool_conv = self._parse_conv_layer()
        gpool_bn = self._parse_batchnorm_layer()
        gpool_activation = self._parse_activation_layer(model_version)
        gpool_to_bias_mul = self._parse_matmul_layer()
        mid_bn = self._parse_batchnorm_layer()
        mid_activation = self._parse_activation_layer(model_version)
        final_conv = self._parse_conv_layer()

        return GlobalPoolingResidualBlockDesc(
            name=name,
            model_version=model_version,
            pre_bn=pre_bn,
            pre_activation=pre_activation,
            regular_conv=regular_conv,
            gpool_conv=gpool_conv,
            gpool_bn=gpool_bn,
            gpool_activation=gpool_activation,
            gpool_to_bias_mul=gpool_to_bias_mul,
            mid_bn=mid_bn,
            mid_activation=mid_activation,
            final_conv=final_conv
        )

    def _parse_nested_bottleneck_block(self, model_version: int, trunk_num_channels: int) -> NestedBottleneckResidualBlockDesc:
        """Parse a nested bottleneck residual block."""
        name = self._read_string()
        num_blocks = self._read_int()

        pre_bn = self._parse_batchnorm_layer()
        pre_activation = self._parse_activation_layer(model_version)
        pre_conv = self._parse_conv_layer()

        blocks = self._parse_block_stack(model_version, num_blocks, pre_conv.out_channels)

        post_bn = self._parse_batchnorm_layer()
        post_activation = self._parse_activation_layer(model_version)
        post_conv = self._parse_conv_layer()

        return NestedBottleneckResidualBlockDesc(
            name=name,
            num_blocks=num_blocks,
            pre_bn=pre_bn,
            pre_activation=pre_activation,
            pre_conv=pre_conv,
            blocks=blocks,
            post_bn=post_bn,
            post_activation=post_activation,
            post_conv=post_conv
        )

    def _parse_block_stack(self, model_version: int, num_blocks: int, trunk_num_channels: int) -> List[Tuple[int, object]]:
        """Parse a stack of residual blocks."""
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

    def _parse_sgf_metadata_encoder(self, model_version: int, meta_encoder_version: int) -> SGFMetadataEncoderDesc:
        """Parse SGF metadata encoder."""
        name = self._read_string()
        num_input_meta_channels = self._read_int()

        mul1 = self._parse_matmul_layer()
        bias1 = self._parse_matbias_layer()
        act1 = self._parse_activation_layer(model_version)
        mul2 = self._parse_matmul_layer()
        bias2 = self._parse_matbias_layer()
        act2 = self._parse_activation_layer(model_version)
        mul3 = self._parse_matmul_layer()

        return SGFMetadataEncoderDesc(
            name=name,
            meta_encoder_version=meta_encoder_version,
            num_input_meta_channels=num_input_meta_channels,
            mul1=mul1,
            bias1=bias1,
            act1=act1,
            mul2=mul2,
            bias2=bias2,
            act2=act2,
            mul3=mul3
        )

    def _parse_trunk(self, model_version: int, meta_encoder_version: int) -> TrunkDesc:
        """Parse the trunk network."""
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

        # Parse SGF metadata encoder if present
        sgf_metadata_encoder = None
        if meta_encoder_version > 0:
            sgf_metadata_encoder = self._parse_sgf_metadata_encoder(model_version, meta_encoder_version)

        # Parse residual blocks
        blocks = self._parse_block_stack(model_version, num_blocks, trunk_num_channels)

        trunk_tip_bn = self._parse_batchnorm_layer()
        trunk_tip_activation = self._parse_activation_layer(model_version)

        return TrunkDesc(
            name=name,
            model_version=model_version,
            num_blocks=num_blocks,
            trunk_num_channels=trunk_num_channels,
            mid_num_channels=mid_num_channels,
            regular_num_channels=regular_num_channels,
            gpool_num_channels=gpool_num_channels,
            meta_encoder_version=meta_encoder_version,
            initial_conv=initial_conv,
            initial_matmul=initial_matmul,
            sgf_metadata_encoder=sgf_metadata_encoder,
            blocks=blocks,
            trunk_tip_bn=trunk_tip_bn,
            trunk_tip_activation=trunk_tip_activation
        )

    def _parse_policy_head(self, model_version: int) -> PolicyHeadDesc:
        """Parse the policy head."""
        name = self._read_string()

        p1_conv = self._parse_conv_layer()
        g1_conv = self._parse_conv_layer()
        g1_bn = self._parse_batchnorm_layer()
        g1_activation = self._parse_activation_layer(model_version)
        gpool_to_bias_mul = self._parse_matmul_layer()
        p1_bn = self._parse_batchnorm_layer()
        p1_activation = self._parse_activation_layer(model_version)
        p2_conv = self._parse_conv_layer()
        gpool_to_pass_mul = self._parse_matmul_layer()

        # Version >= 15 has additional pass move layers
        gpool_to_pass_bias = None
        pass_activation = None
        gpool_to_pass_mul2 = None
        if model_version >= 15:
            gpool_to_pass_bias = self._parse_matbias_layer()
            pass_activation = self._parse_activation_layer(model_version)
            gpool_to_pass_mul2 = self._parse_matmul_layer()

        # Determine policy output channels based on version
        if model_version >= 16:
            policy_out_channels = 4
        elif model_version >= 12:
            policy_out_channels = 2
        else:
            policy_out_channels = 1

        return PolicyHeadDesc(
            name=name,
            model_version=model_version,
            policy_out_channels=policy_out_channels,
            p1_conv=p1_conv,
            g1_conv=g1_conv,
            g1_bn=g1_bn,
            g1_activation=g1_activation,
            gpool_to_bias_mul=gpool_to_bias_mul,
            p1_bn=p1_bn,
            p1_activation=p1_activation,
            p2_conv=p2_conv,
            gpool_to_pass_mul=gpool_to_pass_mul,
            gpool_to_pass_bias=gpool_to_pass_bias,
            pass_activation=pass_activation,
            gpool_to_pass_mul2=gpool_to_pass_mul2
        )

    def _parse_value_head(self, model_version: int) -> ValueHeadDesc:
        """Parse the value head."""
        name = self._read_string()

        v1_conv = self._parse_conv_layer()
        v1_bn = self._parse_batchnorm_layer()
        v1_activation = self._parse_activation_layer(model_version)
        v2_mul = self._parse_matmul_layer()
        v2_bias = self._parse_matbias_layer()
        v2_activation = self._parse_activation_layer(model_version)
        v3_mul = self._parse_matmul_layer()
        v3_bias = self._parse_matbias_layer()
        sv3_mul = self._parse_matmul_layer()
        sv3_bias = self._parse_matbias_layer()
        v_ownership_conv = self._parse_conv_layer()

        return ValueHeadDesc(
            name=name,
            model_version=model_version,
            v1_conv=v1_conv,
            v1_bn=v1_bn,
            v1_activation=v1_activation,
            v2_mul=v2_mul,
            v2_bias=v2_bias,
            v2_activation=v2_activation,
            v3_mul=v3_mul,
            v3_bias=v3_bias,
            sv3_mul=sv3_mul,
            sv3_bias=sv3_bias,
            v_ownership_conv=v_ownership_conv
        )

    def _parse_model(self) -> KataGoModelDesc:
        """Parse the complete model."""
        # Read header
        name = self._read_string()
        model_version = self._read_int()

        if model_version not in self.SUPPORTED_VERSIONS:
            raise ValueError(
                f"Only KataGo model versions {self.SUPPORTED_VERSIONS} are supported, "
                f"got version {model_version}"
            )

        num_input_channels = self._read_int()
        num_input_global_channels = self._read_int()

        # Parse post-process params (version >= 13)
        post_process_params = ModelPostProcessParams()
        if model_version >= 13:
            post_process_params.td_score_multiplier = self._read_float()
            post_process_params.score_mean_multiplier = self._read_float()
            post_process_params.score_stdev_multiplier = self._read_float()
            post_process_params.lead_multiplier = self._read_float()
            post_process_params.variance_time_multiplier = self._read_float()
            post_process_params.shortterm_value_error_multiplier = self._read_float()
            post_process_params.shortterm_score_error_multiplier = self._read_float()

        # Parse meta encoder version (version >= 15)
        meta_encoder_version = 0
        num_input_meta_channels = 0
        if model_version >= 15:
            meta_encoder_version = self._read_int()
            # Read unused params
            for _ in range(7):
                self._read_int()

            if meta_encoder_version > 0:
                num_input_meta_channels = 192  # SGFMetadata::METADATA_INPUT_NUM_CHANNELS

        # Parse trunk, policy head, value head
        trunk = self._parse_trunk(model_version, meta_encoder_version)
        policy_head = self._parse_policy_head(model_version)
        value_head = self._parse_value_head(model_version)

        # Determine output channel counts
        num_policy_channels = policy_head.policy_out_channels
        num_value_channels = 3  # win, loss, noresult
        if model_version >= 9:
            num_score_value_channels = 6
        elif model_version >= 8:
            num_score_value_channels = 4
        elif model_version >= 4:
            num_score_value_channels = 2
        else:
            num_score_value_channels = 1
        num_ownership_channels = 1

        return KataGoModelDesc(
            name=name,
            sha256="",  # Not computed during parsing
            model_version=model_version,
            num_input_channels=num_input_channels,
            num_input_global_channels=num_input_global_channels,
            num_input_meta_channels=num_input_meta_channels,
            num_policy_channels=num_policy_channels,
            num_value_channels=num_value_channels,
            num_score_value_channels=num_score_value_channels,
            num_ownership_channels=num_ownership_channels,
            meta_encoder_version=meta_encoder_version,
            post_process_params=post_process_params,
            trunk=trunk,
            policy_head=policy_head,
            value_head=value_head
        )
