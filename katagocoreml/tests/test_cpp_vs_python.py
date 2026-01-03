#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""
Integration tests to verify C++ katagocoreml library generates binary-identical
Core ML models compared to the Python coremltools/converters/katago converter.

The Python converter has been validated against KataGo's Eigen backend and serves
as the reference implementation. These tests ensure the C++ port produces equivalent
output.
"""

import subprocess
from pathlib import Path
from typing import Optional

import pytest


# ==============================================================================
# Comparison Utilities
# ==============================================================================


def compare_binary_files(file1: Path, file2: Path) -> tuple[bool, str]:
    """Compare two files byte-by-byte.

    Args:
        file1: Path to first file
        file2: Path to second file

    Returns:
        tuple: (identical: bool, details: str)
    """
    if not file1.exists():
        return False, f"File not found: {file1}"
    if not file2.exists():
        return False, f"File not found: {file2}"

    size1 = file1.stat().st_size
    size2 = file2.stat().st_size

    if size1 != size2:
        return False, f"Size mismatch: {file1.name}={size1} bytes, {file2.name}={size2} bytes"

    # Read and compare in chunks to handle large files
    chunk_size = 8192
    offset = 0

    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            chunk1 = f1.read(chunk_size)
            chunk2 = f2.read(chunk_size)

            if chunk1 != chunk2:
                # Find exact byte position of first difference
                for i, (b1, b2) in enumerate(zip(chunk1, chunk2)):
                    if b1 != b2:
                        diff_offset = offset + i
                        return False, (
                            f"Binary difference at byte {diff_offset}: "
                            f"cpp=0x{b1:02x}, python=0x{b2:02x}"
                        )

            if not chunk1:
                break

            offset += len(chunk1)

    return True, "Files are identical"


def compare_mlpackage(cpp_path: Path, python_path: Path) -> tuple[bool, dict]:
    """Compare two .mlpackage directories.

    Compares:
    - Data/com.apple.CoreML/model.mlmodel (protobuf model specification)
    - Data/com.apple.CoreML/weights/weight.bin (weight blob file)

    Args:
        cpp_path: Path to C++ generated .mlpackage
        python_path: Path to Python generated .mlpackage

    Returns:
        tuple: (all_identical: bool, results: dict)
            results contains per-file comparison details
    """
    files_to_compare = [
        "Data/com.apple.CoreML/model.mlmodel",
        "Data/com.apple.CoreML/weights/weight.bin",
    ]

    results = {}
    all_identical = True

    for rel_path in files_to_compare:
        cpp_file = cpp_path / rel_path
        python_file = python_path / rel_path

        identical, details = compare_binary_files(cpp_file, python_file)
        results[rel_path] = {"identical": identical, "details": details}

        if not identical:
            all_identical = False

    return all_identical, results


# ==============================================================================
# Converter Wrappers
# ==============================================================================


def convert_with_cpp(
    exe_path: Path,
    model_path: Path,
    output_path: Path,
    board_size: int,
    optimize_mask: bool,
) -> None:
    """Convert KataGo model using C++ CLI tool.

    Args:
        exe_path: Path to katago2coreml executable
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag

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
) -> None:
    """Convert KataGo model using Python coremltools converter.

    Args:
        model_path: Path to input .bin.gz model
        output_path: Path for output .mlpackage
        board_size: Board size (e.g., 9, 13, 19)
        optimize_mask: Enable optimize_identity_mask flag
    """
    import coremltools as ct
    from coremltools.converters.katago import convert

    mlmodel = convert(
        str(model_path),
        board_x_size=board_size,
        board_y_size=board_size,
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT32,
        optimize_identity_mask=optimize_mask,
    )

    mlmodel.save(str(output_path))


# ==============================================================================
# Test Classes
# ==============================================================================


class TestCppVsPythonConverter:
    """Compare C++ and Python converter outputs for binary equivalence."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "g170e-b10c128-s1141046784-d204142634.bin.gz",  # Standard model
            "g170-b6c96-s175395328-d26788732.bin.gz",       # Smaller model
        ],
    )
    @pytest.mark.parametrize("board_size", [9, 13, 19])
    @pytest.mark.parametrize("optimize_mask", [False, True])
    def test_binary_identical(
        self,
        model_name: str,
        board_size: int,
        optimize_mask: bool,
        katago2coreml_exe: Path,
        all_test_models: dict,
        temp_output_dir: Path,
    ):
        """Test that C++ and Python converters produce binary-identical outputs.

        This test:
        1. Converts a KataGo model using the C++ CLI tool
        2. Converts the same model using the Python converter
        3. Compares model.mlmodel and weight.bin byte-by-byte
        4. Fails if any difference is detected

        Args:
            model_name: Name of the KataGo model file
            board_size: Board size (9, 13, or 19)
            optimize_mask: Whether to enable optimize_identity_mask
            katago2coreml_exe: Path to C++ CLI tool (fixture)
            all_test_models: Dict mapping model names to paths (fixture)
            temp_output_dir: Temporary directory for outputs (fixture)
        """
        # Skip if model not available
        if model_name not in all_test_models:
            pytest.skip(f"Model not available: {model_name}")

        model_path = all_test_models[model_name]

        # Generate unique output names based on configuration
        mask_suffix = "mask_true" if optimize_mask else "mask_false"
        cpp_output = temp_output_dir / f"cpp_{board_size}x{board_size}_{mask_suffix}.mlpackage"
        python_output = temp_output_dir / f"python_{board_size}x{board_size}_{mask_suffix}.mlpackage"

        # Convert with C++
        convert_with_cpp(
            katago2coreml_exe,
            model_path,
            cpp_output,
            board_size,
            optimize_mask,
        )

        # Convert with Python
        convert_with_python(
            model_path,
            python_output,
            board_size,
            optimize_mask,
        )

        # Compare outputs
        identical, results = compare_mlpackage(cpp_output, python_output)

        # Build detailed error message if not identical
        if not identical:
            error_lines = [
                f"Binary mismatch for {model_name} at {board_size}x{board_size} "
                f"(optimize_mask={optimize_mask}):",
            ]
            for file_path, result in results.items():
                status = "OK" if result["identical"] else "DIFFERENT"
                error_lines.append(f"  {file_path}: {status}")
                if not result["identical"]:
                    error_lines.append(f"    {result['details']}")

            pytest.fail("\n".join(error_lines))


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
