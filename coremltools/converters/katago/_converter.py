# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Main converter for KataGo models to Core ML format.
"""

from ._katago_model_builder import KataGoModelBuilder
from ._katago_parser import KataGoModelParser


def convert(
    model_path: str,
    minimum_deployment_target=None,
    compute_precision=None,
    compute_units=None,
    eliminate_identity_mask: bool = False,
):
    """
    Convert a KataGo model to Core ML format.

    Parameters
    ----------
    model_path : str
        Path to KataGo model file (.bin, .bin.gz, or .txt).

    minimum_deployment_target : coremltools.target, optional
        Minimum deployment target. Defaults to iOS15/macOS12.
        For best performance, use ct.target.iOS18 (1.4% faster).

    compute_precision : coremltools.precision, optional
        Compute precision. Defaults to FLOAT32.
        For Neural Engine optimization, use ct.precision.FLOAT16.

    compute_units : coremltools.ComputeUnit, optional
        Compute units to use. Defaults to ALL.

    eliminate_identity_mask : bool, optional
        If True, eliminate mask operations for fixed 19x19 board size.

        This optimization provides ~6.5% inference speedup by precomputing
        mask-derived constants and eliminating mask operations. However, it is
        ONLY valid for full 19x19 board inference where all mask values are 1.0.

        **Important**: Do NOT use with partial boards or variable board sizes.
        The optimization will produce incorrect results if any mask values are 0.

        Default is False (safe for all board configurations).

    Returns
    -------
    coremltools.models.MLModel
        Converted Core ML model.

    Examples
    --------
    Basic conversion (compatible with all board configurations):

    >>> import coremltools as ct
    >>> mlmodel = ct.converters.katago.convert(
    ...     "kata1-b40c256.bin.gz",
    ...     minimum_deployment_target=ct.target.iOS18,
    ...     compute_precision=ct.precision.FLOAT16
    ... )
    >>> mlmodel.save("KataGo.mlpackage")

    Optimized for full 19x19 boards only (6.5% faster):

    >>> mlmodel = ct.converters.katago.convert(
    ...     "kata1-b40c256.bin.gz",
    ...     minimum_deployment_target=ct.target.iOS18,
    ...     compute_precision=ct.precision.FLOAT16,
    ...     eliminate_identity_mask=True
    ... )
    >>> mlmodel.save("KataGo-optimized.mlpackage")

    Notes
    -----
    The converted model has the following inputs:
    - spatial_input: Float32 tensor (1, num_input_channels, 19, 19)
    - global_input: Float32 tensor (1, num_input_global_channels)
    - input_mask: Float32 tensor (1, 1, 19, 19)

    And the following outputs:
    - policy: Float32 tensor (1, policy_channels, 19, 19) - move policy logits
    - pass_policy: Float32 tensor (1, 1) - pass move logit
    - value: Float32 tensor (1, 3) - win/loss/noresult probabilities
    - ownership: Float32 tensor (1, 1, 19, 19) - territory ownership
    - score_value: Float32 tensor (1, score_channels) - score distribution
    """
    import coremltools as ct
    from coremltools import __version__ as ct_version

    # Set default for compute_units
    if compute_units is None:
        compute_units = ct.ComputeUnit.ALL

    # Parse KataGo model
    parser = KataGoModelParser(model_path)
    model_desc = parser.parse()

    # Build MIL program
    builder = KataGoModelBuilder(
        model_desc,
        eliminate_identity_mask=eliminate_identity_mask
    )
    prog = builder.build()

    # Set default deployment target
    if minimum_deployment_target is None:
        minimum_deployment_target = ct.target.iOS15

    # Convert MIL program to Core ML model
    mlmodel = ct.convert(
        prog,
        source="milinternal",
        convert_to="mlprogram",
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=compute_precision,
        compute_units=compute_units,
    )

    # Add metadata
    mlmodel.user_defined_metadata[ct.models._METADATA_VERSION] = ct_version
    mlmodel.user_defined_metadata[ct.models._METADATA_SOURCE] = f"katago=={model_desc.model_version}"
    mlmodel.short_description = f"KataGo model: {model_desc.name}"

    # Add input/output descriptions
    spec = mlmodel.get_spec()

    # Update input descriptions
    for inp in spec.description.input:
        if inp.name == "spatial_input":
            inp.shortDescription = "Board position features (NCHW format)"
        elif inp.name == "global_input":
            inp.shortDescription = "Global game state features"
        elif inp.name == "input_mask":
            inp.shortDescription = "Valid board positions mask"

    # Update output descriptions
    for out in spec.description.output:
        if "policy" in out.name and "pass" not in out.name:
            out.shortDescription = "Move policy logits (NCHW format)"
        elif "pass" in out.name:
            out.shortDescription = "Pass move logit"
        elif "value" in out.name and "score" not in out.name:
            out.shortDescription = "Win/loss/noresult probabilities"
        elif "ownership" in out.name:
            out.shortDescription = "Territory ownership map"
        elif "score" in out.name:
            out.shortDescription = "Score distribution outputs"

    # Save updated spec
    mlmodel = ct.models.MLModel(spec, weights_dir=mlmodel.weights_dir)

    return mlmodel
