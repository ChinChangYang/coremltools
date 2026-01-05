#  Copyright (c) 2025, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Pytest fixtures for katagocoreml C++ vs Python converter cross-validation tests."""

from pathlib import Path

import pytest


# ==============================================================================
# Path Configuration
# ==============================================================================

# Repository root (katagocoreml is at KataGoCoremltools/katagocoreml)
REPO_ROOT = Path(__file__).parent.parent.parent

# C++ CLI tool location
KATAGO2COREML_BUILD_PATH = REPO_ROOT / "katagocoreml" / "build" / "katago2coreml"

# KataGo model locations
KATAGO_MODELS_DIR = REPO_ROOT / "KataGo" / "cpp" / "tests" / "models"


# ==============================================================================
# C++ CLI Tool Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def katago2coreml_exe():
    """Path to the built katago2coreml CLI tool.

    Returns:
        Path: Path to the katago2coreml executable

    Skips:
        If the executable is not found (not built)
    """
    if not KATAGO2COREML_BUILD_PATH.exists():
        pytest.skip(
            f"katago2coreml executable not found: {KATAGO2COREML_BUILD_PATH}. "
            "Build it first: cd katagocoreml/build && cmake .. && make -j"
        )

    return KATAGO2COREML_BUILD_PATH


# ==============================================================================
# Model Fixtures
# ==============================================================================


@pytest.fixture(scope="session")
def standard_model_bin():
    """Path to standard KataGo model (g170e-b10c128).

    Returns:
        Path: Path to the model file

    Skips:
        If the model file is not found
    """
    model_path = KATAGO_MODELS_DIR / "g170e-b10c128-s1141046784-d204142634.bin.gz"

    if not model_path.exists():
        pytest.skip(f"Standard model not found: {model_path}")

    return model_path


@pytest.fixture(scope="session")
def smaller_model_bin():
    """Path to smaller KataGo model (g170-b6c96).

    Returns:
        Path: Path to the model file

    Skips:
        If the model file is not found
    """
    model_path = KATAGO_MODELS_DIR / "g170-b6c96-s175395328-d26788732.bin.gz"

    if not model_path.exists():
        pytest.skip(f"Smaller model not found: {model_path}")

    return model_path


@pytest.fixture(scope="session")
def distilled_model_bin():
    """Path to distilled KataGo model (b5c192nbt) with metadata encoder.

    This is a human SL model that requires meta_input (192 channels).

    Returns:
        Path: Path to the model file

    Skips:
        If the model file is not found
    """
    model_path = KATAGO_MODELS_DIR / "b5c192nbt-distilled.bin.gz"

    if not model_path.exists():
        pytest.skip(f"Distilled model not found: {model_path}")

    return model_path


@pytest.fixture(scope="session")
def all_test_models(standard_model_bin, smaller_model_bin, distilled_model_bin):
    """Dictionary of all available test models.

    Returns:
        dict: Mapping from model name to path
    """
    return {
        "g170e-b10c128-s1141046784-d204142634.bin.gz": standard_model_bin,
        "g170-b6c96-s175395328-d26788732.bin.gz": smaller_model_bin,
        "b5c192nbt-distilled.bin.gz": distilled_model_bin,
    }


# ==============================================================================
# Output Directory Fixtures
# ==============================================================================


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Temporary directory for test outputs.

    This creates a fresh directory for each test function to avoid conflicts.

    Args:
        tmp_path: Pytest's built-in tmp_path fixture

    Returns:
        Path: Path to the temporary directory
    """
    return tmp_path
