# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Data classes for KataGo model layer descriptors.

These classes represent the structure of a KataGo neural network model
as defined in the KataGo source code (cpp/neuralnet/desc.h).
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Union

import numpy as np


class ActivationType(IntEnum):
    """Activation function types used in KataGo models."""
    IDENTITY = 0
    RELU = 1
    MISH = 2
    # MISH_SCALE8 = 12 is internal optimization, treated as MISH


@dataclass
class ConvLayerDesc:
    """
    Convolutional layer descriptor.

    Attributes:
        name: Layer name.
        conv_y_size: Filter height.
        conv_x_size: Filter width.
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dilation_y: Dilation factor in Y direction.
        dilation_x: Dilation factor in X direction.
        weights: Weight tensor in OIHW format (out_channels, in_channels, height, width).
    """
    name: str
    conv_y_size: int
    conv_x_size: int
    in_channels: int
    out_channels: int
    dilation_y: int
    dilation_x: int
    weights: np.ndarray  # Shape: [out_channels, in_channels, y, x] (OIHW format for Core ML)


@dataclass
class BatchNormLayerDesc:
    """
    Batch normalization layer descriptor.

    KataGo pre-computes merged scale and bias for efficiency:
        merged_scale = scale / sqrt(variance + epsilon)
        merged_bias = bias - mean * merged_scale

    During inference: output = input * merged_scale + merged_bias

    Attributes:
        name: Layer name.
        num_channels: Number of channels.
        epsilon: Small constant for numerical stability.
        has_scale: Whether the layer has learnable scale (gamma).
        has_bias: Whether the layer has learnable bias (beta).
        mean: Running mean values.
        variance: Running variance values.
        scale: Scale (gamma) values.
        bias: Bias (beta) values.
        merged_scale: Pre-computed scale for efficient inference.
        merged_bias: Pre-computed bias for efficient inference.
    """
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
    """
    Activation layer descriptor.

    Attributes:
        name: Layer name.
        activation_type: Type of activation function.
    """
    name: str
    activation_type: ActivationType


@dataclass
class MatMulLayerDesc:
    """
    Matrix multiplication (fully connected) layer descriptor.

    Computes: output = input @ weights

    Attributes:
        name: Layer name.
        in_channels: Input dimension.
        out_channels: Output dimension.
        weights: Weight matrix of shape [in_channels, out_channels].
    """
    name: str
    in_channels: int
    out_channels: int
    weights: np.ndarray  # Shape: [in_channels, out_channels]


@dataclass
class MatBiasLayerDesc:
    """
    Bias addition layer descriptor.

    Computes: output = input + bias

    Attributes:
        name: Layer name.
        num_channels: Number of channels/dimensions.
        weights: Bias values.
    """
    name: str
    num_channels: int
    weights: np.ndarray  # Shape: [num_channels]


@dataclass
class ResidualBlockDesc:
    """
    Standard residual block descriptor.

    Architecture:
        input -> preBN -> preActivation -> regularConv ->
                 midBN -> midActivation -> finalConv -> + input

    Attributes:
        name: Block name.
        pre_bn: Pre-activation batch normalization.
        pre_activation: Pre-activation function.
        regular_conv: First convolution.
        mid_bn: Middle batch normalization.
        mid_activation: Middle activation function.
        final_conv: Second convolution.
    """
    name: str
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    regular_conv: ConvLayerDesc
    mid_bn: BatchNormLayerDesc
    mid_activation: ActivationLayerDesc
    final_conv: ConvLayerDesc


@dataclass
class GlobalPoolingResidualBlockDesc:
    """
    Global pooling residual block descriptor.

    Similar to ResidualBlock but includes a global pooling path that
    aggregates spatial information and adds it back as a bias.

    Architecture:
        input -> preBN -> preActivation -> [regularConv, gpoolConv]
        gpoolConv -> gpoolBN -> gpoolActivation -> globalPool -> gpoolToBiasMul
        regularConv + gpoolBias -> midBN -> midActivation -> finalConv -> + input

    Attributes:
        name: Block name.
        model_version: KataGo model version.
        pre_bn: Pre-activation batch normalization.
        pre_activation: Pre-activation function.
        regular_conv: Main path convolution.
        gpool_conv: Global pooling path convolution.
        gpool_bn: Global pooling path batch normalization.
        gpool_activation: Global pooling path activation.
        gpool_to_bias_mul: Projects global features to channel biases.
        mid_bn: Middle batch normalization.
        mid_activation: Middle activation function.
        final_conv: Final convolution.
    """
    name: str
    model_version: int
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    regular_conv: ConvLayerDesc
    gpool_conv: ConvLayerDesc
    gpool_bn: BatchNormLayerDesc
    gpool_activation: ActivationLayerDesc
    gpool_to_bias_mul: MatMulLayerDesc
    mid_bn: BatchNormLayerDesc
    mid_activation: ActivationLayerDesc
    final_conv: ConvLayerDesc


@dataclass
class NestedBottleneckResidualBlockDesc:
    """
    Nested bottleneck residual block descriptor.

    A bottleneck block that can contain other blocks inside it.

    Attributes:
        name: Block name.
        num_blocks: Number of nested blocks.
        pre_bn: Pre-activation batch normalization.
        pre_activation: Pre-activation function.
        pre_conv: Bottleneck reduction convolution.
        blocks: List of (block_kind, block_desc) tuples.
        post_bn: Post batch normalization.
        post_activation: Post activation function.
        post_conv: Bottleneck expansion convolution.
    """
    name: str
    num_blocks: int
    pre_bn: BatchNormLayerDesc
    pre_activation: ActivationLayerDesc
    pre_conv: ConvLayerDesc
    blocks: List  # List of (block_kind, block_desc) tuples
    post_bn: BatchNormLayerDesc
    post_activation: ActivationLayerDesc
    post_conv: ConvLayerDesc


@dataclass
class SGFMetadataEncoderDesc:
    """
    SGF metadata encoder descriptor (model version >= 15).

    Encodes game metadata (komi, handicap, time, etc.) through a 3-layer MLP.

    Attributes:
        name: Encoder name.
        meta_encoder_version: Version of the metadata encoder.
        num_input_meta_channels: Number of input metadata channels.
        mul1: First linear layer.
        bias1: First bias layer.
        act1: First activation.
        mul2: Second linear layer.
        bias2: Second bias layer.
        act2: Second activation.
        mul3: Third linear layer (output).
    """
    name: str
    meta_encoder_version: int
    num_input_meta_channels: int
    mul1: MatMulLayerDesc
    bias1: MatBiasLayerDesc
    act1: ActivationLayerDesc
    mul2: MatMulLayerDesc
    bias2: MatBiasLayerDesc
    act2: ActivationLayerDesc
    mul3: MatMulLayerDesc


# Block kind constants (matching KataGo's desc.h)
ORDINARY_BLOCK_KIND = 0
GLOBAL_POOLING_BLOCK_KIND = 2
NESTED_BOTTLENECK_BLOCK_KIND = 3

# Type alias for block descriptors
BlockDesc = Union[ResidualBlockDesc, GlobalPoolingResidualBlockDesc, NestedBottleneckResidualBlockDesc]


@dataclass
class TrunkDesc:
    """
    Trunk (backbone) network descriptor.

    The trunk processes the spatial input through a series of residual blocks.

    Attributes:
        name: Trunk name.
        model_version: KataGo model version.
        num_blocks: Number of residual blocks.
        trunk_num_channels: Number of channels in the trunk.
        mid_num_channels: Number of channels in residual block mid convolutions.
        regular_num_channels: Number of channels in gpool block regular convolutions.
        gpool_num_channels: Number of channels in gpool block gpool convolutions.
        meta_encoder_version: SGF metadata encoder version (0 = none).
        initial_conv: Initial convolution from input to trunk channels.
        initial_matmul: Projects global input features to trunk channels.
        sgf_metadata_encoder: Optional SGF metadata encoder.
        blocks: List of (block_kind, block_desc) tuples.
        trunk_tip_bn: Final batch normalization.
        trunk_tip_activation: Final activation.
    """
    name: str
    model_version: int
    num_blocks: int
    trunk_num_channels: int
    mid_num_channels: int
    regular_num_channels: int
    gpool_num_channels: int
    meta_encoder_version: int
    initial_conv: ConvLayerDesc
    initial_matmul: MatMulLayerDesc
    sgf_metadata_encoder: Optional[SGFMetadataEncoderDesc]
    blocks: List  # List of (block_kind, block_desc) tuples
    trunk_tip_bn: BatchNormLayerDesc
    trunk_tip_activation: ActivationLayerDesc


@dataclass
class PolicyHeadDesc:
    """
    Policy head descriptor.

    Produces move probability distribution over the board.

    Attributes:
        name: Head name.
        model_version: KataGo model version.
        policy_out_channels: Number of output policy channels.
        p1_conv: First policy convolution.
        g1_conv: Global path convolution.
        g1_bn: Global path batch normalization.
        g1_activation: Global path activation.
        gpool_to_bias_mul: Projects global features to spatial biases.
        p1_bn: Policy batch normalization.
        p1_activation: Policy activation.
        p2_conv: Final policy convolution.
        gpool_to_pass_mul: Projects global features to pass move logit.
        gpool_to_pass_bias: Pass move bias (version >= 15).
        pass_activation: Pass move activation (version >= 15).
        gpool_to_pass_mul2: Second pass move projection (version >= 15).
    """
    name: str
    model_version: int
    policy_out_channels: int
    p1_conv: ConvLayerDesc
    g1_conv: ConvLayerDesc
    g1_bn: BatchNormLayerDesc
    g1_activation: ActivationLayerDesc
    gpool_to_bias_mul: MatMulLayerDesc
    p1_bn: BatchNormLayerDesc
    p1_activation: ActivationLayerDesc
    p2_conv: ConvLayerDesc
    gpool_to_pass_mul: MatMulLayerDesc
    gpool_to_pass_bias: Optional[MatBiasLayerDesc]
    pass_activation: Optional[ActivationLayerDesc]
    gpool_to_pass_mul2: Optional[MatMulLayerDesc]


@dataclass
class ValueHeadDesc:
    """
    Value head descriptor.

    Produces value estimates (win probability, score, ownership).

    Attributes:
        name: Head name.
        model_version: KataGo model version.
        v1_conv: First value convolution.
        v1_bn: Value batch normalization.
        v1_activation: Value activation.
        v2_mul: Second value linear layer.
        v2_bias: Second value bias.
        v2_activation: Second value activation.
        v3_mul: Third value linear layer (outputs win/loss/noresult).
        v3_bias: Third value bias.
        sv3_mul: Score value linear layer.
        sv3_bias: Score value bias.
        v_ownership_conv: Ownership map convolution.
    """
    name: str
    model_version: int
    v1_conv: ConvLayerDesc
    v1_bn: BatchNormLayerDesc
    v1_activation: ActivationLayerDesc
    v2_mul: MatMulLayerDesc
    v2_bias: MatBiasLayerDesc
    v2_activation: ActivationLayerDesc
    v3_mul: MatMulLayerDesc
    v3_bias: MatBiasLayerDesc
    sv3_mul: MatMulLayerDesc
    sv3_bias: MatBiasLayerDesc
    v_ownership_conv: ConvLayerDesc


@dataclass
class ModelPostProcessParams:
    """
    Post-processing parameters for model outputs.

    Attributes:
        td_score_multiplier: TD score multiplier (default 20.0).
        score_mean_multiplier: Score mean multiplier (default 20.0).
        score_stdev_multiplier: Score stdev multiplier (default 20.0).
        lead_multiplier: Lead multiplier (default 20.0).
        variance_time_multiplier: Variance time multiplier (default 40.0).
        shortterm_value_error_multiplier: Shortterm value error multiplier (default 0.25).
        shortterm_score_error_multiplier: Shortterm score error multiplier (default 30.0).
        output_scale_multiplier: Output scale multiplier (default 1.0).
    """
    td_score_multiplier: float = 20.0
    score_mean_multiplier: float = 20.0
    score_stdev_multiplier: float = 20.0
    lead_multiplier: float = 20.0
    variance_time_multiplier: float = 40.0
    shortterm_value_error_multiplier: float = 0.25
    shortterm_score_error_multiplier: float = 30.0
    output_scale_multiplier: float = 1.0


@dataclass
class KataGoModelDesc:
    """
    Complete KataGo model descriptor.

    Attributes:
        name: Model name/identifier.
        sha256: SHA256 hash of the model file.
        model_version: KataGo model version (supported: 16).
        num_input_channels: Number of spatial input channels.
        num_input_global_channels: Number of global input channels.
        num_input_meta_channels: Number of metadata input channels.
        num_policy_channels: Number of policy output channels.
        num_value_channels: Number of value channels.
        num_score_value_channels: Number of score value output channels.
        num_ownership_channels: Number of ownership output channels.
        meta_encoder_version: SGF metadata encoder version.
        post_process_params: Post-processing parameters.
        trunk: Trunk network descriptor.
        policy_head: Policy head descriptor.
        value_head: Value head descriptor.
    """
    name: str
    sha256: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    num_input_meta_channels: int
    num_policy_channels: int
    num_value_channels: int
    num_score_value_channels: int
    num_ownership_channels: int
    meta_encoder_version: int
    post_process_params: ModelPostProcessParams
    trunk: TrunkDesc
    policy_head: PolicyHeadDesc
    value_head: ValueHeadDesc
