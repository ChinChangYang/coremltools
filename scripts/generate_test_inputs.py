#!/usr/bin/env python3
"""Generate test inputs for KataGo to Core ML cross-validation.

This script generates various test cases as JSON files that can be used
to test both the KataGo C++ Eigen backend and the Core ML converted model.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


# Constants matching KataGo v7 model
NUM_SPATIAL_CHANNELS = 22
NUM_GLOBAL_CHANNELS = 19
BOARD_SIZE = 19


def generate_zeros_test() -> dict:
    """Generate all-zeros test case (simplest baseline)."""
    return {
        "name": "zeros",
        "description": "All zeros - simplest baseline case",
        "spatial_input": np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_random_seed_42_test() -> dict:
    """Generate reproducible random test case."""
    np.random.seed(42)
    return {
        "name": "random_seed_42",
        "description": "Reproducible random inputs (seed=42)",
        "spatial_input": (np.random.randn(NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE).astype(np.float32) * 0.1).tolist(),
        "global_input": (np.random.randn(NUM_GLOBAL_CHANNELS).astype(np.float32) * 0.1).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_corner_stone_test() -> dict:
    """Generate single stone in corner test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Channel 1 is typically player stones
    spatial[1, 0, 0] = 1.0
    # Channel 0 is typically the valid board mask
    spatial[0, :, :] = 1.0

    return {
        "name": "corner_stone",
        "description": "Single player stone at corner (0,0)",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_center_stone_test() -> dict:
    """Generate single stone in center test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    center = BOARD_SIZE // 2
    # Channel 1 is typically player stones
    spatial[1, center, center] = 1.0
    # Channel 0 is typically the valid board mask
    spatial[0, :, :] = 1.0

    return {
        "name": "center_stone",
        "description": "Single player stone at center (9,9)",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_partial_mask_9x9_test() -> dict:
    """Generate partial board mask (9x9 in corner) test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Set valid board mask for 9x9 region
    spatial[0, :9, :9] = 1.0

    mask = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    mask[:9, :9] = 1.0

    return {
        "name": "partial_mask_9x9",
        "description": "9x9 board mask in top-left corner",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": mask.tolist(),
    }


def generate_edge_pattern_test() -> dict:
    """Generate stones along edge test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0
    # Channel 1 is player stones - place along top edge
    for i in range(0, BOARD_SIZE, 2):
        spatial[1, 0, i] = 1.0
    # Channel 2 is opponent stones - place along second row
    for i in range(1, BOARD_SIZE, 2):
        spatial[2, 1, i] = 1.0

    return {
        "name": "edge_pattern",
        "description": "Alternating stones along top edge",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_diagonal_pattern_test() -> dict:
    """Generate diagonal pattern test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0
    # Channel 1 is player stones - diagonal
    for i in range(BOARD_SIZE):
        spatial[1, i, i] = 1.0

    return {
        "name": "diagonal_pattern",
        "description": "Diagonal line of player stones",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_uniform_small_values_test() -> dict:
    """Generate uniform small values test case."""
    spatial = np.full((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), 0.01, dtype=np.float32)
    global_in = np.full(NUM_GLOBAL_CHANNELS, 0.01, dtype=np.float32)

    return {
        "name": "uniform_small",
        "description": "Uniform small values (0.01) for all inputs",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def generate_komi_test() -> dict:
    """Generate test with komi value set."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    spatial[0, :, :] = 1.0  # Valid board mask

    global_in = np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32)
    # Set komi-related features (approximation, actual mapping depends on model version)
    global_in[5] = 7.5 / 20.0  # Normalized komi

    return {
        "name": "komi_7_5",
        "description": "Empty board with komi = 7.5",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((BOARD_SIZE, BOARD_SIZE), dtype=np.float32).tolist(),
    }


def save_test_case(test_case: dict, output_dir: Path) -> None:
    """Save a test case to JSON file."""
    name = test_case["name"]
    filepath = output_dir / f"{name}.json"

    with open(filepath, "w") as f:
        json.dump(test_case, f, indent=2)

    print(f"  Created: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test inputs for KataGo to Core ML cross-validation"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="test_inputs",
        help="Output directory for test input JSON files (default: test_inputs)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test inputs in {output_dir}/")

    # Generate all test cases
    test_generators = [
        generate_zeros_test,
        generate_random_seed_42_test,
        generate_corner_stone_test,
        generate_center_stone_test,
        generate_partial_mask_9x9_test,
        generate_edge_pattern_test,
        generate_diagonal_pattern_test,
        generate_uniform_small_values_test,
        generate_komi_test,
    ]

    for generator in test_generators:
        test_case = generator()
        save_test_case(test_case, output_dir)

    print(f"\nGenerated {len(test_generators)} test cases")

    # Also create an index file
    index = {
        "description": "KataGo to Core ML cross-validation test inputs",
        "board_size": BOARD_SIZE,
        "num_spatial_channels": NUM_SPATIAL_CHANNELS,
        "num_global_channels": NUM_GLOBAL_CHANNELS,
        "test_cases": [gen().__getitem__("name") for gen in test_generators]
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Created index: {index_path}")


if __name__ == "__main__":
    main()
