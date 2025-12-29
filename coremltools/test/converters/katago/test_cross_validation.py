#  Copyright (c) 2024, Apple Inc. All rights reserved.
#
#  Use of this source code is governed by a BSD-3-clause license that can be
#  found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

"""Cross-validation tests for KataGo Core ML converter.

This test suite validates that Core ML model outputs match KataGo Eigen backend
outputs across multiple board sizes (9x9, 13x13, 19x19) and various input patterns.
"""

import numpy as np
import pytest

from .validation_utils import (
    compare_outputs,
    get_default_tolerances,
    load_test_input,
    run_coreml_model,
    run_eigen_backend_cached,
)


# Mark all tests in this module as slow (model conversion + inference)
pytestmark = pytest.mark.slow


class TestKataGoCrossValidation:
    """Cross-validation tests for KataGo Core ML converter.

    Tests validate that Core ML model outputs match KataGo Eigen backend
    outputs across multiple board sizes (9x9, 13x13, 19x19) and various
    input patterns.
    """

    @pytest.mark.parametrize(
        "test_case",
        [
            "zeros",
            "random_seed_42",
            "corner_stone",
            "center_stone",
            "edge_pattern",
            "diagonal_pattern",
            "uniform_small",
            "komi_7_5",
        ],
    )
    def test_cross_validation_against_eigen(
        self,
        board_size,
        converted_model,
        test_inputs_dir,
        katago_model_bin,
        katago_executable,
        check_eigen_backend_available,
        eigen_cache_dir,
        test_case,
    ):
        """Test Core ML model against Eigen backend for specific test case.

        This test:
        1. Loads test input for the specified test case
        2. Runs inference on both Core ML and Eigen backend
        3. Compares outputs with predefined tolerances
        4. Asserts that all outputs match within tolerance

        Args:
            board_size: Board size (9, 13, or 19) from parametrized fixture
            converted_model: Path to converted Core ML model
            test_inputs_dir: Directory containing test inputs for this board size
            katago_model_bin: Path to KataGo binary model
            katago_executable: Path to KataGo executable
            check_eigen_backend_available: Fixture that skips if Eigen unavailable
            eigen_cache_dir: Directory for caching Eigen backend outputs
            test_case: Test case name (e.g., "zeros", "random_seed_42")
        """
        # Load test input
        test_file = test_inputs_dir / f"{test_case}.json"
        if not test_file.exists():
            pytest.skip(f"Test case '{test_case}' not found for {board_size}x{board_size}")

        inputs = load_test_input(test_file)

        # Run Core ML model
        coreml_out = run_coreml_model(converted_model, inputs)

        # Run Eigen backend (with caching)
        eigen_out = run_eigen_backend_cached(
            katago_model_bin, inputs, katago_executable,
            eigen_cache_dir, test_case, board_size
        )
        if eigen_out is None:
            pytest.fail("Eigen backend execution failed")

        # Compare outputs with default tolerances
        tolerances = get_default_tolerances()
        results = compare_outputs(eigen_out, coreml_out, tolerances)

        # Assert all outputs pass
        failures = []
        for key, result in results.items():
            status = result["status"]
            if status != "PASS":
                failures.append(
                    f"{key}: {status} - max_diff={result.get('max_diff', 'N/A')}, "
                    f"tolerance={result.get('tolerance', 'N/A')}"
                )

        if failures:
            failure_msg = (
                f"Cross-validation failed for {test_case} on {board_size}x{board_size}:\n"
                + "\n".join(f"  - {f}" for f in failures)
            )
            pytest.fail(failure_msg)

    def test_partial_mask(
        self,
        board_size,
        converted_model,
        test_inputs_dir,
        katago_model_bin,
        katago_executable,
        check_eigen_backend_available,
        eigen_cache_dir,
    ):
        """Test with partial board mask (board size dependent).

        Partial mask test case name varies by board size:
        - 9x9: partial_mask_4x4
        - 13x13: partial_mask_6x6
        - 19x19: partial_mask_9x9

        Args:
            board_size: Board size (9, 13, or 19) from parametrized fixture
            converted_model: Path to converted Core ML model
            test_inputs_dir: Directory containing test inputs for this board size
            katago_model_bin: Path to KataGo binary model
            katago_executable: Path to KataGo executable
            check_eigen_backend_available: Fixture that skips if Eigen unavailable
            eigen_cache_dir: Directory for caching Eigen backend outputs
        """
        # Determine partial mask test case name based on board size
        mask_size = board_size // 2
        test_case = f"partial_mask_{mask_size}x{mask_size}"

        test_file = test_inputs_dir / f"{test_case}.json"
        if not test_file.exists():
            pytest.skip(f"Partial mask test case not found for {board_size}x{board_size}")

        inputs = load_test_input(test_file)

        # Run both backends (with caching for Eigen)
        coreml_out = run_coreml_model(converted_model, inputs)
        eigen_out = run_eigen_backend_cached(
            katago_model_bin, inputs, katago_executable,
            eigen_cache_dir, test_case, board_size
        )
        if eigen_out is None:
            pytest.fail("Eigen backend execution failed")

        # Compare outputs with default tolerances
        results = compare_outputs(eigen_out, coreml_out)

        # Assert all outputs pass
        failures = []
        for key, result in results.items():
            status = result["status"]
            if status != "PASS":
                failures.append(
                    f"{key}: {status} - max_diff={result.get('max_diff', 'N/A')}, "
                    f"mean_diff={result.get('mean_diff', 'N/A')}, "
                    f"tolerance={result.get('tolerance', 'N/A')}"
                )

        if failures:
            failure_msg = (
                f"Partial mask test failed on {board_size}x{board_size}:\n"
                + "\n".join(f"  - {f}" for f in failures)
            )
            pytest.fail(failure_msg)


class TestKataGoCorMLOnly:
    """Tests that run Core ML inference without Eigen comparison.

    These tests verify Core ML model can run inference and produce
    valid outputs, even when Eigen backend is not available.
    """

    @pytest.mark.slow
    def test_coreml_inference_smoke_test(self, board_size, converted_model, test_inputs_dir):
        """Smoke test: verify Core ML model can run inference.

        This test loads the zeros test case and runs inference to verify:
        - Model can execute without errors
        - All expected outputs are present
        - No NaN or Inf values in outputs

        Args:
            board_size: Board size (9, 13, or 19) from parametrized fixture
            converted_model: Path to converted Core ML model
            test_inputs_dir: Directory containing test inputs for this board size
        """
        test_file = test_inputs_dir / "zeros.json"
        inputs = load_test_input(test_file)

        outputs = run_coreml_model(converted_model, inputs)

        # Verify outputs exist and have valid shapes
        assert "policy" in outputs, "policy output missing"
        assert "pass_policy" in outputs, "pass_policy output missing"
        assert "value" in outputs, "value output missing"
        assert "ownership" in outputs, "ownership output missing"
        assert "score_value" in outputs, "score_value output missing"

        # Verify no NaN/Inf values
        for key, value in outputs.items():
            assert not np.isnan(value).any(), f"{key} contains NaN values"
            assert not np.isinf(value).any(), f"{key} contains Inf values"

    @pytest.mark.slow
    def test_output_shapes(self, board_size, converted_model, test_inputs_dir):
        """Test output shapes match expected dimensions for board size.

        This test verifies that all model outputs have the correct shapes
        for the given board size.

        Args:
            board_size: Board size (9, 13, or 19) from parametrized fixture
            converted_model: Path to converted Core ML model
            test_inputs_dir: Directory containing test inputs for this board size
        """
        test_file = test_inputs_dir / "zeros.json"
        inputs = load_test_input(test_file)

        outputs = run_coreml_model(converted_model, inputs)

        # Expected shapes
        expected_shapes = {
            "policy": (board_size, board_size),
            "pass_policy": (1,),
            "value": (3,),
            "ownership": (board_size, board_size),
            "score_value": (6,),
        }

        for key, expected_shape in expected_shapes.items():
            actual_shape = outputs[key].shape
            assert actual_shape == expected_shape, (
                f"{key} shape mismatch for {board_size}x{board_size}: "
                f"expected {expected_shape}, got {actual_shape}"
            )
