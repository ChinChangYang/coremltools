#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Integration tests to verify C++ katagocoreml library generates inference-equivalent
Core ML models compared to the Python coremltools/converters/katago converter.

The Python converter has been validated against KataGo's Eigen backend and serves
as the reference implementation. These tests ensure the C++ port produces models
with equivalent inference results within acceptable numerical tolerance.
"""

import subprocess
from pathlib import Path

import pytest


# ==============================================================================
# Converter Wrappers
# ==============================================================================


def convert_with_cpp(
    exe_path: Path,
    model_path: Path,
    output_path: Path,
    board_size: int,
    optimize_mask: bool,
    float16: bool = False,
) -> None:
    """Convert KataGo model using C++ CLI tool.

    Args:
        exe_path: Path to katago2coreml executable
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag
        float16: Enable FLOAT16 precision

    Raises:
        subprocess.CalledProcessError: If conversion fails
    """
    cmd = [
        str(exe_path),
        "-x", str(board_size),
        "-y", str(board_size),
    ]

    if optimize_mask:
        cmd.append("--optimize-identity-mask")

    if float16:
        cmd.append("--float16")

    cmd.extend([str(model_path), str(output_path)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )


def convert_with_python(
    model_path: Path,
    output_path: Path,
    board_size: int,
    optimize_mask: bool,
    float16: bool = False,
) -> None:
    """Convert KataGo model using Python coremltools converter.

    Args:
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag
        float16: Enable FLOAT16 precision
    """
    import coremltools as ct
    from coremltools.converters.katago import convert

    precision = ct.precision.FLOAT16 if float16 else ct.precision.FLOAT32

    mlmodel = convert(
        str(model_path),
        board_x_size=board_size,
        board_y_size=board_size,
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=precision,
        optimize_identity_mask=optimize_mask,
    )

    mlmodel.save(str(output_path))


# ==============================================================================
# Test Classes
# ==============================================================================


class TestCppVsPythonConverterFP32:
    """Tests for C++ FP32 converter producing inference-equivalent models.

    Since binary equivalence between C++ and Python converters is difficult to achieve
    (due to metadata differences, protobuf field ordering, and operation serialization),
    we instead validate that both converters produce models with equivalent inference results.

    This approach validates what actually matters: that the models produce the same outputs
    for the same inputs within acceptable numerical tolerance.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170e-b10c128-s1141046784-d204142634.bin.gz",  # Standard model
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 13, 19])
    @pytest.mark.parametrize("optimize_mask", [False, True])
    def test_fp32_inference_equivalent(
        self,
        model_name: str,
        board_size: int,
        optimize_mask: bool,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that C++ and Python FP32 models produce equivalent inference results.

        This test:
        1. Converts a KataGo model using both C++ and Python converters (FP32)
        2. Loads both models using coremltools
        3. Generates deterministic random inputs
        4. Runs inference on both models
        5. Compares outputs with FP32 tolerance

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9, 13, or 19)
            optimize_mask: Whether to enable optimize_identity_mask
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.processor() != "arm":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Generate unique output names based on configuration
        mask_suffix = "mask_true" if optimize_mask else "mask_false"
        cpp_output = temp_output_dir / f"cpp_fp32_{board_size}x{board_size}_{mask_suffix}.mlpackage"
        python_output = temp_output_dir / f"python_fp32_{board_size}x{board_size}_{mask_suffix}.mlpackage"

        # Convert with both converters
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            cpp_output,
            board_size,
            optimize_mask,
            float16=False,
        )

        convert_with_python(
            model_path,
            python_output,
            board_size,
            optimize_mask,
            float16=False,
        )

        # Load models
        import coremltools as ct
        cpp_model = ct.models.MLModel(str(cpp_output))
        python_model = ct.models.MLModel(str(python_output))

        # Generate deterministic random input
        np.random.seed(42)
        spatial_input = np.random.randn(1, 22, board_size, board_size).astype(np.float32)
        global_input = np.random.randn(1, 19).astype(np.float32)
        input_mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

        inputs = {
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask,
        }

        # Add meta_input for human SL networks (models with metadata encoder)
        cpp_spec = cpp_model.get_spec()
        cpp_input_names = [inp.name for inp in cpp_spec.description.input]
        if "meta_input" in cpp_input_names:
            inputs["meta_input"] = np.zeros((1, 192), dtype=np.float32)

        # Run inference
        cpp_outputs = cpp_model.predict(inputs)
        python_outputs = python_model.predict(inputs)

        # Compare outputs with relative tolerance (matching cross-validation approach)
        # Use same tolerance strategy as validation_utils.py for consistency
        # Map output names to tolerance categories
        output_tolerances = {
            "policy_p2_conv": 0.015,       # policy: 1.5% relative
            "policy_pass_mul2": 0.015,     # pass_policy
            "policy_pass": 0.015,          # pass_policy (version-dependent name)
            "value_v3_bias": 0.01,         # value: 1% relative
            "value_ownership_conv": 0.03,  # ownership: 3% relative
            "value_sv3_bias": 0.015,       # score_value: 1.5% relative
        }
        default_tolerance = 0.03  # 3% for unrecognized outputs

        failed_outputs = []
        for key in python_outputs.keys():
            if key not in cpp_outputs:
                failed_outputs.append(f"Missing output key in C++ model: {key}")
                continue

            cpp_val = cpp_outputs[key]
            py_val = python_outputs[key]

            # Compute relative error: max_diff / max(abs(reference))
            max_diff = np.max(np.abs(cpp_val - py_val))
            max_ref = np.max(np.abs(py_val))
            if max_ref > 1e-8:
                rel_error = max_diff / max_ref
            else:
                rel_error = max_diff  # Absolute if reference is near zero

            tolerance = output_tolerances.get(key, default_tolerance)
            if rel_error > tolerance:
                failed_outputs.append(
                    f"Output '{key}': rel_error={rel_error:.4f} > tolerance={tolerance}"
                )

        if failed_outputs:
            pytest.fail("\n".join(failed_outputs))


class TestConverterSmoke:
    """Smoke tests for individual converter functionality."""

    def test_cpp_converter_help(self, katago2coreml_exe: Path):
        """Test that C++ CLI tool shows help without error."""
        result = subprocess.run(
            [str(katago2coreml_exe), "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "katago2coreml" in result.stdout or "Usage" in result.stdout

    def test_cpp_converter_model_info(
        self,
        katago2coreml_exe: Path,
        standard_model_bin: Path,
    ):
        """Test that C++ CLI can read model info."""
        result = subprocess.run(
            [str(katago2coreml_exe), "--info", str(standard_model_bin)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Should output model information
        assert "version" in result.stdout.lower() or "blocks" in result.stdout.lower()

    def test_python_converter_import(self):
        """Test that Python converter can be imported."""
        try:
            from coremltools.converters.katago import convert
            assert callable(convert)
        except ImportError as e:
            pytest.skip(f"coremltools KataGo converter not available: {e}")


class TestCppVsPythonConverterFP16:
    """Tests for C++ FLOAT16 converter functionality.

    The C++ converter now implements the same approach as Python for FLOAT16:
    1. Model inputs remain FLOAT32
    2. Cast operations convert inputs to FLOAT16 for internal processing
    3. All intermediate operations use FLOAT16 weights and activations
    4. Cast operations convert outputs back to FLOAT32
    5. Weights are stored as FLOAT16 in the blob file

    This matches Python's `add_fp16_cast` pass behavior, producing models
    that Core ML can compile and execute with FP16 precision.
    """

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
            "b5c192nbt-distilled.bin.gz",                   # Distilled model (human SL with metadata)
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 19])
    def test_fp16_inference_equivalent(
        self,
        model_name: str,
        board_size: int,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that C++ and Python FLOAT16 models produce equivalent inference results.

        This test verifies that the C++ converter with cast operations at input/output
        boundaries produces models that Core ML can compile and that produce inference
        results equivalent to the Python converter within FP16 tolerance.

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9 or 19)
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        import platform
        if platform.processor() != "arm":
            pytest.skip("Core ML inference only available on Apple Silicon")

        import numpy as np

        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Convert with both converters
        cpp_output = temp_output_dir / f"cpp_fp16_{board_size}x{board_size}.mlpackage"
        python_output = temp_output_dir / f"python_fp16_{board_size}x{board_size}.mlpackage"

        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            cpp_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        convert_with_python(
            model_path,
            python_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        # Load models
        import coremltools as ct
        cpp_model = ct.models.MLModel(str(cpp_output))
        python_model = ct.models.MLModel(str(python_output))

        # Generate random input
        np.random.seed(42)
        spatial_input = np.random.randn(1, 22, board_size, board_size).astype(np.float32)
        global_input = np.random.randn(1, 19).astype(np.float32)
        input_mask = np.ones((1, 1, board_size, board_size), dtype=np.float32)

        inputs = {
            "spatial_input": spatial_input,
            "global_input": global_input,
            "input_mask": input_mask,
        }

        # Add meta_input for human SL networks (models with metadata encoder)
        cpp_spec = cpp_model.get_spec()
        cpp_input_names = [inp.name for inp in cpp_spec.description.input]
        if "meta_input" in cpp_input_names:
            inputs["meta_input"] = np.zeros((1, 192), dtype=np.float32)

        # Run inference
        cpp_outputs = cpp_model.predict(inputs)
        python_outputs = python_model.predict(inputs)

        # Compare outputs with FP16 tolerance
        # FP16 has lower precision than FP32, so numerical differences up to ~0.5 are acceptable
        tolerance = 0.5
        for key in python_outputs.keys():
            if key not in cpp_outputs:
                pytest.fail(f"Missing output key in C++ model: {key}")

            cpp_val = cpp_outputs[key]
            py_val = python_outputs[key]

            max_diff = np.max(np.abs(cpp_val - py_val))
            if max_diff > tolerance:
                pytest.fail(
                    f"Output '{key}' differs by max {max_diff:.6f} "
                    f"(tolerance: {tolerance})"
                )

    def test_fp16_weight_size_reduced(
        self,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Verify FLOAT16 weight.bin is approximately half the size of FLOAT32.

        This test ensures weights are actually being stored as FP16.
        """
        model_name = "g170-b6c96-s175395328-d26788732.bin.gz"  # Smaller model
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]
        board_size = 9

        # Convert with FLOAT32
        fp32_output = temp_output_dir / "fp32_size_test.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            fp32_output,
            board_size,
            optimize_mask=False,
            float16=False,
        )

        # Convert with FLOAT16
        fp16_output = temp_output_dir / "fp16_size_test.mlpackage"
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            fp16_output,
            board_size,
            optimize_mask=False,
            float16=True,
        )

        # Compare weight file sizes
        fp32_weights = fp32_output / "Data/com.apple.CoreML/weights/weight.bin"
        fp16_weights = fp16_output / "Data/com.apple.CoreML/weights/weight.bin"

        fp32_size = fp32_weights.stat().st_size
        fp16_size = fp16_weights.stat().st_size

        # FP16 weights should be roughly half the size of FP32
        # Allow some tolerance for blob header overhead
        expected_ratio = 0.5
        actual_ratio = fp16_size / fp32_size

        assert 0.45 < actual_ratio < 0.55, (
            f"FLOAT16 weight size ratio unexpected: {actual_ratio:.3f} "
            f"(expected ~{expected_ratio}). "
            f"FP32: {fp32_size} bytes, FP16: {fp16_size} bytes"
        )
