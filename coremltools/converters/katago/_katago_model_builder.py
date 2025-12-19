# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
MIL program builder for KataGo models.

This module constructs a complete MIL program from a parsed KataGo model.
"""

import numpy as np

from coremltools.converters.mil.mil import Builder as mb, types

from ._katago_ops import KataGoOps
from ._katago_types import (
    GLOBAL_POOLING_BLOCK_KIND,
    NESTED_BOTTLENECK_BLOCK_KIND,
    ORDINARY_BLOCK_KIND,
    GlobalPoolingResidualBlockDesc,
    KataGoModelDesc,
    NestedBottleneckResidualBlockDesc,
    ResidualBlockDesc,
)


class KataGoModelBuilder:
    """
    Builder for constructing MIL programs from KataGo models.

    Converts a parsed KataGo model description into a MIL program that can
    be converted to Core ML format.
    """

    BOARD_SIZE = 19

    def __init__(self, model_desc: KataGoModelDesc):
        """
        Initialize the model builder.

        Args:
            model_desc: Parsed KataGo model description.
        """
        self.model_desc = model_desc
        self.ops = KataGoOps()

    def build(self):
        """
        Build and return the MIL program.

        Returns:
            A MIL program that can be converted to Core ML.
        """
        num_input_ch = self.model_desc.num_input_channels
        num_global_ch = self.model_desc.num_input_global_channels

        # Define input specs
        input_specs = [
            mb.TensorSpec(shape=(1, num_input_ch, self.BOARD_SIZE, self.BOARD_SIZE), dtype=types.fp32),
            mb.TensorSpec(shape=(1, num_global_ch), dtype=types.fp32),
            mb.TensorSpec(shape=(1, 1, self.BOARD_SIZE, self.BOARD_SIZE), dtype=types.fp32),
        ]

        # Build the program using decorator
        @mb.program(input_specs=input_specs)
        def katago_model(spatial_input, global_input, input_mask):
            return self._build_model(spatial_input, global_input, input_mask)

        return katago_model

    def _build_model(self, spatial_input, global_input, input_mask):
        """
        Build the complete KataGo model.

        Args:
            spatial_input: Spatial input tensor [N, C, H, W].
            global_input: Global input tensor [N, G].
            input_mask: Mask tensor [N, 1, H, W].

        Returns:
            Tuple of output tensors (policy, pass_policy, value, ownership, score_value).
        """
        # Build trunk
        trunk_out = self._build_trunk(spatial_input, global_input, input_mask)

        # Build policy head
        policy, pass_policy = self._build_policy_head(trunk_out, input_mask)

        # Build value head
        value, ownership, score_value = self._build_value_head(trunk_out, input_mask)

        return policy, pass_policy, value, ownership, score_value

    def _build_trunk(self, spatial_input, global_input, input_mask):
        """
        Build the trunk (backbone) network.

        Args:
            spatial_input: Spatial input tensor.
            global_input: Global input tensor.
            input_mask: Mask tensor.

        Returns:
            Trunk output tensor.
        """
        trunk = self.model_desc.trunk

        # Initial conv: spatial input -> trunk channels
        x = self.ops.build_conv(spatial_input, trunk.initial_conv, name="trunk_initial_conv")

        # Compute mask sum features for global input processing
        mask_sum, mask_sum_sqrt_s14_m01, mask_sum_sqrt_s14_m01_sq_s01 = \
            self.ops.build_mask_sum_features(input_mask, name="trunk")

        # Reshape global input for matmul: [N, G] -> [N, G]
        # Append mask sum features to global input
        mask_sum_squeezed = mb.squeeze(x=mask_sum, axes=[2, 3], name="trunk_mask_sum_squeeze")
        mask_sqrt_squeezed = mb.squeeze(x=mask_sum_sqrt_s14_m01, axes=[2, 3], name="trunk_mask_sqrt_squeeze")
        mask_sq_squeezed = mb.squeeze(x=mask_sum_sqrt_s14_m01_sq_s01, axes=[2, 3], name="trunk_mask_sq_squeeze")

        global_input_extended = mb.concat(
            values=[global_input, mask_sum_squeezed, mask_sqrt_squeezed, mask_sq_squeezed],
            axis=1,
            name="trunk_global_concat"
        )

        # Project global features to trunk channels
        global_bias = self.ops.build_matmul(global_input_extended, trunk.initial_matmul, name="trunk_initial_matmul")

        # Reshape to [N, trunk_ch, 1, 1] for broadcasting
        global_bias = mb.reshape(x=global_bias, shape=[1, -1, 1, 1], name="trunk_global_reshape")

        # Add global bias to spatial features
        x = mb.add(x=x, y=global_bias, name="trunk_add_global")

        # Apply mask
        x = mb.mul(x=x, y=input_mask, name="trunk_initial_mask")

        # Process residual blocks
        for i, (block_kind, block) in enumerate(trunk.blocks):
            if block_kind == ORDINARY_BLOCK_KIND:
                x = self._build_residual_block(x, block, input_mask, f"trunk_block_{i}")
            elif block_kind == GLOBAL_POOLING_BLOCK_KIND:
                x = self._build_global_pooling_residual_block(x, block, input_mask, f"trunk_gpool_block_{i}")
            elif block_kind == NESTED_BOTTLENECK_BLOCK_KIND:
                x = self._build_nested_bottleneck_block(x, block, input_mask, f"trunk_nested_block_{i}")

        # Trunk tip: BN + activation
        x = self.ops.build_batchnorm_activation(
            x, trunk.trunk_tip_bn, trunk.trunk_tip_activation, input_mask, name="trunk_tip"
        )

        return x

    def _build_residual_block(self, x, block: ResidualBlockDesc, mask, name: str):
        """
        Build a standard residual block.

        Architecture: input -> BN -> Act -> Conv -> BN -> Act -> Conv -> + input
        """
        residual = x

        # Pre-BN + activation
        x = self.ops.build_batchnorm_activation(x, block.pre_bn, block.pre_activation, mask, f"{name}_pre")

        # First conv
        x = self.ops.build_conv(x, block.regular_conv, f"{name}_conv1")

        # Mid-BN + activation
        x = self.ops.build_batchnorm_activation(x, block.mid_bn, block.mid_activation, mask, f"{name}_mid")

        # Second conv
        x = self.ops.build_conv(x, block.final_conv, f"{name}_conv2")

        # Residual connection
        x = mb.add(x=x, y=residual, name=f"{name}_residual")

        return x

    def _build_global_pooling_residual_block(self, x, block: GlobalPoolingResidualBlockDesc, mask, name: str):
        """
        Build a global pooling residual block.

        Similar to residual block but includes a global pooling path.
        """
        residual = x

        # Pre-BN + activation
        x = self.ops.build_batchnorm_activation(x, block.pre_bn, block.pre_activation, mask, f"{name}_pre")

        # Regular conv path
        regular_out = self.ops.build_conv(x, block.regular_conv, f"{name}_regular_conv")

        # Global pooling path
        gpool_out = self.ops.build_conv(x, block.gpool_conv, f"{name}_gpool_conv")
        gpool_out = self.ops.build_batchnorm_activation(
            gpool_out, block.gpool_bn, block.gpool_activation, mask, f"{name}_gpool"
        )

        # Global pooling: mean + max + mean*sqrt (produces 3x channels)
        gpool_features = self.ops.build_global_pooling(gpool_out, mask, f"{name}_global_pool")

        # Project global features to bias
        gpool_bias = self.ops.build_matmul(gpool_features, block.gpool_to_bias_mul, f"{name}_gpool_bias")
        gpool_bias = mb.reshape(x=gpool_bias, shape=[1, -1, 1, 1], name=f"{name}_gpool_bias_reshape")

        # Add global bias to regular path
        x = mb.add(x=regular_out, y=gpool_bias, name=f"{name}_add_gpool_bias")

        # Mid-BN + activation
        x = self.ops.build_batchnorm_activation(x, block.mid_bn, block.mid_activation, mask, f"{name}_mid")

        # Final conv
        x = self.ops.build_conv(x, block.final_conv, f"{name}_final_conv")

        # Residual connection
        x = mb.add(x=x, y=residual, name=f"{name}_residual")

        return x

    def _build_nested_bottleneck_block(self, x, block: NestedBottleneckResidualBlockDesc, mask, name: str):
        """
        Build a nested bottleneck residual block.
        """
        residual = x

        # Pre-BN + activation + bottleneck reduction
        x = self.ops.build_batchnorm_activation(x, block.pre_bn, block.pre_activation, mask, f"{name}_pre")
        x = self.ops.build_conv(x, block.pre_conv, f"{name}_pre_conv")

        # Process nested blocks
        for i, (block_kind, nested_block) in enumerate(block.blocks):
            if block_kind == ORDINARY_BLOCK_KIND:
                x = self._build_residual_block(x, nested_block, mask, f"{name}_nested_{i}")
            elif block_kind == GLOBAL_POOLING_BLOCK_KIND:
                x = self._build_global_pooling_residual_block(x, nested_block, mask, f"{name}_nested_gpool_{i}")

        # Post-BN + activation + bottleneck expansion
        x = self.ops.build_batchnorm_activation(x, block.post_bn, block.post_activation, mask, f"{name}_post")
        x = self.ops.build_conv(x, block.post_conv, f"{name}_post_conv")

        # Residual connection
        x = mb.add(x=x, y=residual, name=f"{name}_residual")

        return x

    def _build_policy_head(self, trunk_out, mask):
        """
        Build the policy head.

        Returns:
            Tuple of (policy_logits, pass_logit).
        """
        policy_head = self.model_desc.policy_head

        # P1 conv path
        p1 = self.ops.build_conv(trunk_out, policy_head.p1_conv, "policy_p1_conv")

        # G1 conv path (for global pooling)
        g1 = self.ops.build_conv(trunk_out, policy_head.g1_conv, "policy_g1_conv")
        g1 = self.ops.build_batchnorm_activation(
            g1, policy_head.g1_bn, policy_head.g1_activation, mask, "policy_g1"
        )

        # Global pooling on g1
        g1_pooled = self.ops.build_global_pooling(g1, mask, "policy_g1_pool")

        # Project global features to spatial bias
        gpool_bias = self.ops.build_matmul(g1_pooled, policy_head.gpool_to_bias_mul, "policy_gpool_to_bias")
        gpool_bias = mb.reshape(x=gpool_bias, shape=[1, -1, 1, 1], name="policy_gpool_bias_reshape")

        # Add global bias to p1
        p1 = mb.add(x=p1, y=gpool_bias, name="policy_add_gpool_bias")

        # P1 BN + activation
        p1 = self.ops.build_batchnorm_activation(
            p1, policy_head.p1_bn, policy_head.p1_activation, mask, "policy_p1"
        )

        # P2 conv to get policy logits
        policy = self.ops.build_conv(p1, policy_head.p2_conv, "policy_p2_conv")

        # Compute pass move logit
        if policy_head.gpool_to_pass_mul2 is not None:
            # Version >= 15: two-layer pass computation
            pass_hidden = self.ops.build_matmul(g1_pooled, policy_head.gpool_to_pass_mul, "policy_pass_mul1")
            pass_hidden = self.ops.build_matbias(pass_hidden, policy_head.gpool_to_pass_bias, "policy_pass_bias")
            pass_hidden = self.ops.build_activation(pass_hidden, policy_head.pass_activation, "policy_pass_act")
            pass_logit = self.ops.build_matmul(pass_hidden, policy_head.gpool_to_pass_mul2, "policy_pass_mul2")
        else:
            # Version < 15: single layer pass computation
            pass_logit = self.ops.build_matmul(g1_pooled, policy_head.gpool_to_pass_mul, "policy_pass")

        return policy, pass_logit

    def _build_value_head(self, trunk_out, mask):
        """
        Build the value head.

        Returns:
            Tuple of (value, ownership, score_value).
        """
        value_head = self.model_desc.value_head

        # V1 conv
        v1 = self.ops.build_conv(trunk_out, value_head.v1_conv, "value_v1_conv")
        v1 = self.ops.build_batchnorm_activation(
            v1, value_head.v1_bn, value_head.v1_activation, mask, "value_v1"
        )

        # Global pooling on v1
        v1_pooled = self.ops.build_global_pooling(v1, mask, "value_v1_pool")

        # V2: linear + bias + activation
        v2 = self.ops.build_matmul(v1_pooled, value_head.v2_mul, "value_v2_mul")
        v2 = self.ops.build_matbias(v2, value_head.v2_bias, "value_v2_bias")
        v2 = self.ops.build_activation(v2, value_head.v2_activation, "value_v2_act")

        # V3: linear + bias -> value output (win/loss/noresult probabilities)
        value = self.ops.build_matmul(v2, value_head.v3_mul, "value_v3_mul")
        value = self.ops.build_matbias(value, value_head.v3_bias, "value_v3_bias")

        # SV3: score value output
        score_value = self.ops.build_matmul(v2, value_head.sv3_mul, "value_sv3_mul")
        score_value = self.ops.build_matbias(score_value, value_head.sv3_bias, "value_sv3_bias")

        # Ownership conv
        ownership = self.ops.build_conv(v1, value_head.v_ownership_conv, "value_ownership_conv")

        return value, ownership, score_value
