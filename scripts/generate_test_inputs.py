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


def generate_zeros_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate empty board test case (simplest realistic baseline)."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 0 should always be the valid board mask
    spatial[0, :, :] = 1.0

    return {
        "name": "zeros",
        "description": f"Empty board ({board_x_size}x{board_y_size}) - simplest realistic baseline",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_random_seed_42_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate reproducible random binary pattern test case."""
    np.random.seed(42)
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0
    # Randomly place stones in channels 1-6 (binary 0/1)
    for channel in range(1, 7):
        random_mask = np.random.rand(board_y_size, board_x_size) > 0.7
        spatial[channel, random_mask] = 1.0

    global_in = np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32)
    # Set some global features with binary/normalized values
    global_in[0] = 1.0  # Side to move
    global_in[5] = 0.375  # Normalized komi (7.5/20)

    return {
        "name": "random_seed_42",
        "description": f"Reproducible random binary pattern ({board_x_size}x{board_y_size}, seed=42)",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_corner_stone_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate single stone in corner test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 1 is typically player stones
    spatial[1, 0, 0] = 1.0
    # Channel 0 is typically the valid board mask
    spatial[0, :, :] = 1.0

    return {
        "name": "corner_stone",
        "description": f"Single player stone at corner (0,0) on {board_x_size}x{board_y_size} board",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_center_stone_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate single stone in center test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    center_y = board_y_size // 2
    center_x = board_x_size // 2
    # Channel 1 is typically player stones
    spatial[1, center_y, center_x] = 1.0
    # Channel 0 is typically the valid board mask
    spatial[0, :, :] = 1.0

    return {
        "name": "center_stone",
        "description": f"Single player stone at center ({center_y},{center_x}) on {board_x_size}x{board_y_size} board",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_partial_mask_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate partial board mask test case (half of the board masked)."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Set valid board mask for half the board (top half)
    mask_y = board_y_size // 2
    mask_x = board_x_size // 2
    spatial[0, :mask_y, :mask_x] = 1.0

    mask = np.zeros((board_y_size, board_x_size), dtype=np.float32)
    mask[:mask_y, :mask_x] = 1.0

    return {
        "name": f"partial_mask_{mask_x}x{mask_y}",
        "description": f"{mask_x}x{mask_y} board mask in top-left corner of {board_x_size}x{board_y_size} board",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": mask.tolist(),
    }


def generate_edge_pattern_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate stones along edge test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0
    # Channel 1 is player stones - place along top edge
    for i in range(0, board_x_size, 2):
        spatial[1, 0, i] = 1.0
    # Channel 2 is opponent stones - place along second row
    for i in range(1, board_x_size, 2):
        if board_y_size > 1:
            spatial[2, 1, i] = 1.0

    return {
        "name": "edge_pattern",
        "description": f"Alternating stones along top edge ({board_x_size}x{board_y_size})",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_diagonal_pattern_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate diagonal pattern test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0
    # Channel 1 is player stones - diagonal
    for i in range(min(board_x_size, board_y_size)):
        spatial[1, i, i] = 1.0

    return {
        "name": "diagonal_pattern",
        "description": f"Diagonal line of player stones ({board_x_size}x{board_y_size})",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_uniform_small_values_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate complex multi-stone pattern test case."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    # Channel 0 is valid board mask
    spatial[0, :, :] = 1.0

    # Create a complex pattern with multiple stone groups
    # Player stones (channel 1) - fill every 3rd position
    for i in range(0, board_y_size, 3):
        for j in range(0, board_x_size, 3):
            spatial[1, i, j] = 1.0

    # Opponent stones (channel 2) - fill offset positions
    for i in range(1, board_y_size, 3):
        for j in range(1, board_x_size, 3):
            spatial[2, i, j] = 1.0

    global_in = np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32)
    global_in[0] = 1.0  # Side to move

    return {
        "name": "uniform_small",
        "description": f"Complex multi-stone pattern ({board_x_size}x{board_y_size})",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
    }


def generate_komi_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate test with komi value set."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    spatial[0, :, :] = 1.0  # Valid board mask

    global_in = np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32)
    # Set komi-related features (approximation, actual mapping depends on model version)
    global_in[5] = 7.5 / 20.0  # Normalized komi

    return {
        "name": "komi_7_5",
        "description": f"Empty board ({board_x_size}x{board_y_size}) with komi = 7.5",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
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
    parser.add_argument(
        "--board-x-size",
        type=int,
        default=19,
        help="Board width (number of columns, default: 19)"
    )
    parser.add_argument(
        "--board-y-size",
        type=int,
        default=19,
        help="Board height (number of rows, default: 19)"
    )
    args = parser.parse_args()

    board_x_size = args.board_x_size
    board_y_size = args.board_y_size

    # Validate board size
    if not (2 <= board_x_size <= 37):
        raise ValueError(f"board_x_size must be in range [2, 37], got {board_x_size}")
    if not (2 <= board_y_size <= 37):
        raise ValueError(f"board_y_size must be in range [2, 37], got {board_y_size}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test inputs for {board_x_size}x{board_y_size} board in {output_dir}/")

    # Generate all test cases
    test_generators = [
        generate_zeros_test,
        generate_random_seed_42_test,
        generate_corner_stone_test,
        generate_center_stone_test,
        generate_partial_mask_test,
        generate_edge_pattern_test,
        generate_diagonal_pattern_test,
        generate_uniform_small_values_test,
        generate_komi_test,
    ]

    for generator in test_generators:
        test_case = generator(board_x_size, board_y_size)
        save_test_case(test_case, output_dir)

    print(f"\nGenerated {len(test_generators)} test cases")

    # Also create an index file
    index = {
        "description": "KataGo to Core ML cross-validation test inputs",
        "board_x_size": board_x_size,
        "board_y_size": board_y_size,
        "num_spatial_channels": NUM_SPATIAL_CHANNELS,
        "num_global_channels": NUM_GLOBAL_CHANNELS,
        "test_cases": [gen(board_x_size, board_y_size)["name"] for gen in test_generators]
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Created index: {index_path}")


if __name__ == "__main__":
    main()
