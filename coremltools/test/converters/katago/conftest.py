#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest fixtures for KataGo Core ML cross-validation tests."""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def katago_model_bin():
    """Path to KataGo .bin.gz model file.

    Returns:
        str: Path to the KataGo model binary

    Skips:
        If the KataGo model binary is not found
    """
    # Look for model in katago_eigen directory
    model_path = Path(__file__).parent.parent.parent.parent.parent / "KataGo" / "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"

    if not model_path.exists():
        pytest.skip(f"KataGo model not found: {model_path}")

    return str(model_path)


@pytest.fixture(scope="session")
def katago_executable():
    """Path to KataGo executable with validation subcommand.

    Returns:
        str: Path to the KataGo executable

    Skips:
        If the KataGo executable is not found
    """
    # Look for executable in katago_eigen/cpp/build directory
    exe_path = Path(__file__).parent.parent.parent.parent.parent / "KataGo" / "cpp" / "build" / "katago"

    if not exe_path.exists():
        pytest.skip(f"KataGo executable not found: {exe_path}")

    return str(exe_path)


@pytest.fixture(scope="session")
def check_eigen_backend_available(katago_executable):
    """Check if Eigen backend is available and functional.

    Args:
        katago_executable: Path to KataGo executable (from fixture)

    Skips:
        If the Eigen backend is not available or not functional
    """
    from .validation_utils import is_eigen_backend_available

    if not is_eigen_backend_available(katago_executable):
        pytest.skip("KataGo Eigen backend not available or not functional")


@pytest.fixture(scope="session", params=[9, 13, 19])
def board_size(request):
    """Parametrize board sizes: 9x9, 13x13, 19x19.

    Returns:
        int: Board size (9, 13, or 19)
    """
    return request.param


@pytest.fixture(scope="session")
def converted_model(board_size, tmp_path_factory, katago_model_bin):
    """Convert KataGo model to Core ML for given board size (cached per session).

    This fixture converts the model once per board size and reuses it across all
    test cases for that board size.

    Args:
        board_size: Board size (from parametrized fixture)
        tmp_path_factory: Pytest factory for creating temp directories
        katago_model_bin: Path to KataGo binary model (from fixture)

    Returns:
        str: Path to converted .mlpackage model

    Raises:
        ImportError: If coremltools is not installed
    """
    import coremltools as ct
    from coremltools.converters.katago import convert

    # Create session-level temp directory for this board size
    tmp_dir = tmp_path_factory.mktemp(f"models_{board_size}x{board_size}")
    model_path = tmp_dir / f"KataGo_{board_size}x{board_size}.mlpackage"

    # If already converted (shouldn't happen with session scope, but safety check)
    if model_path.exists():
        return str(model_path)

    # Convert model
    mlmodel = convert(
        katago_model_bin,
        board_x_size=board_size,
        board_y_size=board_size,
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT32,
    )
    mlmodel.save(str(model_path))

    return str(model_path)


@pytest.fixture(scope="session")
def test_inputs_dir(board_size):
    """Get test inputs directory for given board size.

    Args:
        board_size: Board size (from parametrized fixture)

    Returns:
        Path: Path to test inputs directory

    Skips:
        If the test inputs directory doesn't exist for this board size
    """
    test_dir = Path(__file__).parent.parent.parent.parent.parent / "test_inputs" / f"{board_size}x{board_size}"

    if not test_dir.exists():
        pytest.skip(
            f"Test inputs not found for {board_size}x{board_size}. "
            f"Run: python scripts/generate_test_inputs.py --output test_inputs/{board_size}x{board_size} "
            f"--board-x-size {board_size} --board-y-size {board_size}"
        )

    return test_dir


@pytest.fixture
def test_input_files(test_inputs_dir):
    """Load all test input JSON files for the board size.

    Args:
        test_inputs_dir: Path to test inputs directory (from fixture)

    Returns:
        list[Path]: List of test input JSON file paths

    Raises:
        pytest.fail: If no test input files are found
    """
    files = sorted(test_inputs_dir.glob("*.json"))
    files = [f for f in files if f.name != "index.json"]

    if not files:
        pytest.fail(f"No test input files found in {test_inputs_dir}")

    return files


@pytest.fixture(scope="session")
def eigen_cache_dir():
    """Get or create Eigen output cache directory.

    Returns:
        Path: Path to the Eigen output cache directory

    Note:
        Cache files are stored with the pattern:
        {model_hash}_{board_size}x{board_size}_{test_case}.json
    """
    cache_dir = Path(__file__).parent.parent.parent.parent.parent / "test_inputs" / "eigen_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir
