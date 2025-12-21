# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
MIL operation builders for KataGo model layers.

This module provides functions to build MIL operations for various
KataGo neural network layer types.
"""

import numpy as np

from coremltools.converters.mil.mil import Builder as mb

from ._katago_types import (
    ActivationLayerDesc,
    ActivationType,
    BatchNormLayerDesc,
    ConvLayerDesc,
    MatBiasLayerDesc,
    MatMulLayerDesc,
)


class KataGoOps:
    """
    Builder for KataGo-specific MIL operations.

    Provides methods to convert KataGo layer descriptors to MIL operations.
    """

    BOARD_SIZE = 19

    def build_conv(self, x, layer: ConvLayerDesc, name: str):
        """
        Build a convolution operation.

        Args:
            x: Input tensor.
            layer: Convolutional layer descriptor.
            name: Operation name.

        Returns:
            Output tensor from the convolution.
        """
        # Weights are already in OIHW format from parser
        weights = layer.weights.astype(np.float32)

        # Calculate padding for "same" behavior with dilation
        # pad_y = (kernel_size - 1) * dilation // 2
        pad_y = (layer.conv_y_size - 1) * layer.dilation_y // 2
        pad_x = (layer.conv_x_size - 1) * layer.dilation_x // 2

        return mb.conv(
            x=x,
            weight=weights,
            pad_type="custom",
            pad=[pad_y, pad_y, pad_x, pad_x],
            dilations=[layer.dilation_y, layer.dilation_x],
            strides=[1, 1],
            groups=1,
            name=name
        )

    def build_batchnorm(self, x, layer: BatchNormLayerDesc, mask, name: str):
        """
        Build batch normalization using pre-merged scale and bias.

        KataGo uses merged scale/bias for efficiency:
            output = (input * merged_scale + merged_bias) * mask

        Args:
            x: Input tensor.
            layer: Batch normalization layer descriptor.
            mask: Mask tensor for zeroing out invalid positions.
            name: Operation name prefix.

        Returns:
            Output tensor after batch normalization and masking.
        """
        # Reshape scale/bias for broadcasting: [C] -> [1, C, 1, 1]
        scale = layer.merged_scale.astype(np.float32).reshape(1, -1, 1, 1)
        bias = layer.merged_bias.astype(np.float32).reshape(1, -1, 1, 1)

        # Apply: x * scale + bias
        x = mb.mul(x=x, y=scale, name=f"{name}_scale")
        x = mb.add(x=x, y=bias, name=f"{name}_bias")

        # Apply mask (zero out padded regions)
        x = mb.mul(x=x, y=mask, name=f"{name}_mask")

        return x

    def build_activation(self, x, layer: ActivationLayerDesc, name: str):
        """
        Build an activation function.

        Args:
            x: Input tensor.
            layer: Activation layer descriptor.
            name: Operation name.

        Returns:
            Output tensor after activation.
        """
        if layer.activation_type == ActivationType.IDENTITY:
            return x
        elif layer.activation_type == ActivationType.RELU:
            return mb.relu(x=x, name=name)
        elif layer.activation_type == ActivationType.MISH:
            return self.build_mish(x, name)
        else:
            raise ValueError(f"Unknown activation type: {layer.activation_type}")

    def build_mish(self, x, name: str):
        """
        Build Mish activation: x * tanh(softplus(x)).

        softplus(x) = log(1 + exp(min(x, threshold)))

        The threshold is used to prevent overflow in exp().
        For float32, threshold=20 is safe (exp(20) ~ 4.8e8).

        Args:
            x: Input tensor.
            name: Operation name prefix.

        Returns:
            Output tensor after Mish activation.
        """
        # Clamp x for softplus stability (prevent exp overflow)
        threshold = np.float32(20.0)
        x_clamped = mb.minimum(x=x, y=threshold, name=f"{name}_clamp")

        # softplus = log(1 + exp(x_clamped))
        exp_x = mb.exp(x=x_clamped, name=f"{name}_exp")
        one_plus_exp = mb.add(x=exp_x, y=np.float32(1.0), name=f"{name}_one_plus_exp")
        softplus = mb.log(x=one_plus_exp, name=f"{name}_softplus")

        # For x > threshold, softplus(x) ≈ x, so use select
        # This ensures numerical correctness for large values
        use_x = mb.greater(x=x, y=threshold, name=f"{name}_use_x")
        softplus = mb.select(cond=use_x, a=x, b=softplus, name=f"{name}_softplus_select")

        # tanh(softplus(x))
        tanh_sp = mb.tanh(x=softplus, name=f"{name}_tanh")

        # x * tanh(softplus(x))
        return mb.mul(x=x, y=tanh_sp, name=name)

    def build_batchnorm_activation(self, x, bn_layer: BatchNormLayerDesc,
                                    act_layer: ActivationLayerDesc, mask, name: str):
        """
        Build combined batch normalization followed by activation.

        Args:
            x: Input tensor.
            bn_layer: Batch normalization layer descriptor.
            act_layer: Activation layer descriptor.
            mask: Mask tensor.
            name: Operation name prefix.

        Returns:
            Output tensor after batch norm and activation.
        """
        x = self.build_batchnorm(x, bn_layer, mask, f"{name}_bn")
        x = self.build_activation(x, act_layer, f"{name}_act")
        return x

    def build_matmul(self, x, layer: MatMulLayerDesc, name: str):
        """
        Build matrix multiplication (linear layer without bias).

        Args:
            x: Input tensor of shape [N, in_channels] or [N, in_channels, 1, 1].
            layer: Matrix multiplication layer descriptor.
            name: Operation name.

        Returns:
            Output tensor of shape [N, out_channels].
        """
        weights = layer.weights.astype(np.float32)
        return mb.matmul(x=x, y=weights, name=name)

    def build_matbias(self, x, layer: MatBiasLayerDesc, name: str):
        """
        Add bias to input.

        Args:
            x: Input tensor.
            layer: Bias layer descriptor.
            name: Operation name.

        Returns:
            Output tensor with bias added.
        """
        bias = layer.weights.astype(np.float32)
        return mb.add(x=x, y=bias, name=name)

    def build_global_pooling(self, x, mask, name: str):
        """
        Build KataGo global pooling for trunk and policy head.

        KataGo poolRowsGPool produces three features in this order:
        1. Mean pooling (sum / count of valid positions)
        2. Mean * (sqrt(count) - 14) * 0.1
        3. Max pooling (maximum over valid positions)

        Args:
            x: Input tensor of shape [N, C, H, W].
            mask: Mask tensor of shape [N, 1, H, W].
            name: Operation name prefix.

        Returns:
            Output tensor of shape [N, C*3] with concatenated pooling results.
        """
        # Count valid (non-masked) positions
        mask_sum = mb.reduce_sum(x=mask, axes=[2, 3], keep_dims=True, name=f"{name}_mask_sum")

        # Mean pooling (masked)
        masked_x = mb.mul(x=x, y=mask, name=f"{name}_masked")
        sum_x = mb.reduce_sum(x=masked_x, axes=[2, 3], keep_dims=True, name=f"{name}_sum")
        mean_x = mb.real_div(x=sum_x, y=mask_sum, name=f"{name}_mean")

        # Max pooling (masked) - set masked positions to large negative value
        neg_inf = np.float32(-1e9)
        inv_mask = mb.sub(x=np.float32(1.0), y=mask, name=f"{name}_inv_mask")
        mask_offset = mb.mul(x=inv_mask, y=neg_inf, name=f"{name}_mask_offset")
        x_for_max = mb.add(x=masked_x, y=mask_offset, name=f"{name}_x_for_max")
        max_x = mb.reduce_max(x=x_for_max, axes=[2, 3], keep_dims=True, name=f"{name}_max")

        # Mean * (sqrt(count) - 14) * 0.1 pooling (correct formula)
        sqrt_mask_sum = mb.sqrt(x=mask_sum, name=f"{name}_sqrt_mask_sum")
        sqrt_minus_14 = mb.sub(x=sqrt_mask_sum, y=np.float32(14.0), name=f"{name}_sqrt_m14")
        scaled_factor = mb.mul(x=sqrt_minus_14, y=np.float32(0.1), name=f"{name}_scaled_factor")
        mean_scaled_x = mb.mul(x=mean_x, y=scaled_factor, name=f"{name}_mean_scaled")

        # Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        mean_flat = mb.squeeze(x=mean_x, axes=[2, 3], name=f"{name}_mean_flat")
        mean_scaled_flat = mb.squeeze(x=mean_scaled_x, axes=[2, 3], name=f"{name}_mean_scaled_flat")
        max_flat = mb.squeeze(x=max_x, axes=[2, 3], name=f"{name}_max_flat")

        # Concatenate in correct order: [mean, mean_scaled, max]
        return mb.concat(
            values=[mean_flat, mean_scaled_flat, max_flat],
            axis=1,
            name=f"{name}_concat"
        )

    def build_global_pooling_value(self, x, mask, name: str):
        """
        Build KataGo global pooling for value head (different from trunk/policy).

        KataGo poolRowsValueHead produces three features in this order:
        1. Mean pooling (sum / count of valid positions)
        2. Mean * (sqrt(count) - 14) * 0.1
        3. Mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)

        Args:
            x: Input tensor of shape [N, C, H, W].
            mask: Mask tensor of shape [N, 1, H, W].
            name: Operation name prefix.

        Returns:
            Output tensor of shape [N, C*3] with concatenated pooling results.
        """
        # Count valid (non-masked) positions
        mask_sum = mb.reduce_sum(x=mask, axes=[2, 3], keep_dims=True, name=f"{name}_mask_sum")

        # Mean pooling (masked)
        masked_x = mb.mul(x=x, y=mask, name=f"{name}_masked")
        sum_x = mb.reduce_sum(x=masked_x, axes=[2, 3], keep_dims=True, name=f"{name}_sum")
        mean_x = mb.real_div(x=sum_x, y=mask_sum, name=f"{name}_mean")

        # Compute (sqrt(count) - 14)
        sqrt_mask_sum = mb.sqrt(x=mask_sum, name=f"{name}_sqrt_mask_sum")
        sqrt_minus_14 = mb.sub(x=sqrt_mask_sum, y=np.float32(14.0), name=f"{name}_sqrt_m14")

        # Feature 2: Mean * (sqrt(count) - 14) * 0.1
        scaled_factor = mb.mul(x=sqrt_minus_14, y=np.float32(0.1), name=f"{name}_scaled_factor")
        mean_scaled_x = mb.mul(x=mean_x, y=scaled_factor, name=f"{name}_mean_scaled")

        # Feature 3: Mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)
        sqrt_m14_sq = mb.mul(x=sqrt_minus_14, y=sqrt_minus_14, name=f"{name}_sqrt_m14_sq")
        sqrt_m14_sq_01 = mb.mul(x=sqrt_m14_sq, y=np.float32(0.01), name=f"{name}_sq_01")
        feature3_factor = mb.sub(x=sqrt_m14_sq_01, y=np.float32(0.1), name=f"{name}_f3_factor")
        mean_feature3_x = mb.mul(x=mean_x, y=feature3_factor, name=f"{name}_mean_f3")

        # Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        mean_flat = mb.squeeze(x=mean_x, axes=[2, 3], name=f"{name}_mean_flat")
        mean_scaled_flat = mb.squeeze(x=mean_scaled_x, axes=[2, 3], name=f"{name}_mean_scaled_flat")
        mean_f3_flat = mb.squeeze(x=mean_feature3_x, axes=[2, 3], name=f"{name}_mean_f3_flat")

        # Concatenate: [mean, mean_scaled, mean_feature3]
        return mb.concat(
            values=[mean_flat, mean_scaled_flat, mean_f3_flat],
            axis=1,
            name=f"{name}_concat"
        )

    def build_mask_sum_features(self, mask, name: str):
        """
        Build mask sum features used for global input processing.

        Computes:
        - mask_sum: sum of valid positions
        - mask_sum_sqrt_s14_m01: (sqrt(mask_sum) - 14) * 0.1
        - mask_sum_sqrt_s14_m01_sq_s01: ((sqrt(mask_sum) - 14) * 0.1)^2 - 0.1

        Args:
            mask: Mask tensor of shape [N, 1, H, W].
            name: Operation name prefix.

        Returns:
            Tuple of (mask_sum, mask_sum_sqrt_s14_m01, mask_sum_sqrt_s14_m01_sq_s01).
        """
        # mask_sum: [N, 1, 1, 1]
        mask_sum = mb.reduce_sum(x=mask, axes=[2, 3], keep_dims=True, name=f"{name}_mask_sum")

        # sqrt(mask_sum)
        sqrt_mask_sum = mb.sqrt(x=mask_sum, name=f"{name}_sqrt_mask_sum")

        # (sqrt(mask_sum) - 14) * 0.1
        fourteen = np.float32(14.0)
        zero_point_one = np.float32(0.1)
        subtracted = mb.sub(x=sqrt_mask_sum, y=fourteen, name=f"{name}_sub_14")
        mask_sum_sqrt_s14_m01 = mb.mul(x=subtracted, y=zero_point_one, name=f"{name}_sqrt_s14_m01")

        # ((sqrt(mask_sum) - 14) * 0.1)^2 - 0.1
        squared = mb.mul(x=mask_sum_sqrt_s14_m01, y=mask_sum_sqrt_s14_m01, name=f"{name}_squared")
        mask_sum_sqrt_s14_m01_sq_s01 = mb.sub(x=squared, y=zero_point_one, name=f"{name}_sq_s01")

        return mask_sum, mask_sum_sqrt_s14_m01, mask_sum_sqrt_s14_m01_sq_s01
