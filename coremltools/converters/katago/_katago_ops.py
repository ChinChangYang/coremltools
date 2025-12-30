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

    def __init__(
        self,
        board_x_size: int = 19,
        board_y_size: int = 19,
        optimize_identity_mask: bool = False,
    ):
        """
        Initialize the KataGoOps builder.

        Args:
            board_x_size: Board width (number of columns). Must be in range [2, 37].
            board_y_size: Board height (number of rows). Must be in range [2, 37].
            optimize_identity_mask: If True, optimize inference by skipping internal mask operations.
                When the mask covers the full board (all mask values are 1.0),
                internal mask multiplications and mask_sum computations can be eliminated/precomputed.
                This optimization provides ~6.5% inference speedup but is only valid for
                full board inference. Do not use with partial boards.

                Important: The input_mask parameter is still required in the model interface.
                This optimization only affects internal operations.
        """
        self.board_x_size = board_x_size
        self.board_y_size = board_y_size
        self.optimize_identity_mask = optimize_identity_mask

        # Precompute mask-derived constants for full board
        if self.optimize_identity_mask:
            self.mask_sum_constant = float(self.board_x_size * self.board_y_size)
            self.mask_sum_reciprocal = 1.0 / self.mask_sum_constant  # Precomputed reciprocal
            sqrt_mask_sum = np.sqrt(self.mask_sum_constant)
            self.mask_sum_sqrt_s14_m01_constant = (sqrt_mask_sum - 14.0) * 0.1
            self.mask_sum_sqrt_s14_m01_sq_s01_constant = (self.mask_sum_sqrt_s14_m01_constant ** 2) - 0.1

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

        return mb.conv(
            x=x,
            weight=weights,
            pad_type="same",
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
        # Note: input_mask is ALWAYS required in model interface
        # This optimization only affects internal operations:
        # - Skips mask multiplications (assumes all mask values = 1.0)
        # - Precomputes mask-derived constants
        # - Does NOT remove input_mask from model signature
        if not self.optimize_identity_mask:
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
        Build Mish activation: x / (1 + 2 / (e * (e + 2))).

        e = exp(x)

        This is KataGo's 6-op exp-based implementation, which provides
        the best balance of accuracy and performance on Apple Neural Engine.

        Args:
            x: Input tensor.
            name: Operation name prefix.

        Returns:
            Output tensor after Mish activation.
        """
        e = mb.exp(x=x)
        ep2 = mb.add(x=e, y=2.0)
        emep2 = mb.mul(x=e, y=ep2)
        tdemep2 = mb.real_div(x=2.0, y=emep2)
        optdemep2 = mb.add(x=1.0, y=tdemep2)
        res = mb.real_div(x=x, y=optdemep2, name=name)

        return res

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
        if self.optimize_identity_mask:
            # Optimized path: all mask values are 1.0
            # Precomputes constants and eliminates mask operations for faster inference
            # Mean pooling = average over all positions
            sum_x = mb.reduce_sum(x=x, axes=[2, 3], keep_dims=True, name=f"{name}_sum")
            mean_x = mb.mul(x=sum_x, y=np.float32(self.mask_sum_reciprocal), name=f"{name}_mean")

            # Max pooling (no mask adjustment needed)
            max_x = mb.reduce_max(x=x, axes=[2, 3], keep_dims=True, name=f"{name}_max")

            # Mean * (sqrt(count) - 14) * 0.1 = mean * 0.5 (precomputed)
            mean_scaled_x = mb.mul(x=mean_x, y=np.float32(self.mask_sum_sqrt_s14_m01_constant), name=f"{name}_mean_scaled")

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
        else:
            # Original path: mask operations included
            # Count valid (non-masked) positions
            mask_sum = mb.reduce_sum(x=mask, axes=[2, 3], keep_dims=True, name=f"{name}_mask_sum")

            # Mean pooling (masked)
            masked_x = mb.mul(x=x, y=mask, name=f"{name}_masked")
            sum_x = mb.reduce_sum(x=masked_x, axes=[2, 3], keep_dims=True, name=f"{name}_sum")
            mean_x = mb.real_div(x=sum_x, y=mask_sum, name=f"{name}_mean")

            # Max pooling (masked) - set masked positions to large negative value
            mask_minus_one = mb.sub(x=mask, y=np.float32(1.0), name=f"{name}_mask_minus_one")
            x_for_max = mb.add(x=masked_x, y=mask_minus_one, name=f"{name}_x_for_max")
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
        if self.optimize_identity_mask:
            # Optimized path: all mask values are 1.0
            # Precomputes constants and eliminates mask operations for faster inference
            # Mean pooling = average over all positions
            sum_x = mb.reduce_sum(x=x, axes=[2, 3], keep_dims=True, name=f"{name}_sum")
            mean_x = mb.mul(x=sum_x, y=np.float32(self.mask_sum_reciprocal), name=f"{name}_mean")

            # Feature 2: Mean * (sqrt(count) - 14) * 0.1 = mean * 0.5 (precomputed)
            mean_scaled_x = mb.mul(x=mean_x, y=np.float32(self.mask_sum_sqrt_s14_m01_constant), name=f"{name}_mean_scaled")

            # Feature 3: Mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1) = mean * 0.15 (precomputed)
            mean_feature3_x = mb.mul(x=mean_x, y=np.float32(self.mask_sum_sqrt_s14_m01_sq_s01_constant), name=f"{name}_mean_f3")

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
        else:
            # Original path: mask operations included
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
