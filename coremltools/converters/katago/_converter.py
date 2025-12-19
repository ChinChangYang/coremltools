# Copyright (c) 2024, Apple Inc. All rights reserved.
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Main converter for KataGo models to Core ML format.
"""

import coremltools as ct
from coremltools import __version__ as ct_version

from ._katago_model_builder import KataGoModelBuilder
from ._katago_parser import KataGoModelParser


def convert(
    model_path: str,
    minimum_deployment_target=None,
    compute_precision=None,
    compute_units=ct.ComputeUnit.ALL,
):
    """
    Convert a KataGo model to Core ML format.

    Parameters
    ----------
    model_path : str
        Path to KataGo model file (.bin, .bin.gz, or .txt).

    minimum_deployment_target : coremltools.target, optional
        Minimum deployment target. Defaults to iOS15/macOS12.

    compute_precision : coremltools.precision, optional
        Compute precision. Defaults to FLOAT32.

    compute_units : coremltools.ComputeUnit, optional
        Compute units to use. Defaults to ALL.

    Returns
    -------
    coremltools.models.MLModel
        Converted Core ML model.

    Examples
    --------
    >>> import coremltools as ct
    >>> mlmodel = ct.converters.katago.convert("kata1-b40c256.bin.gz")
    >>> mlmodel.save("KataGo.mlpackage")

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
    # Parse KataGo model
    parser = KataGoModelParser(model_path)
    model_desc = parser.parse()

    # Build MIL program
    builder = KataGoModelBuilder(model_desc)
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
