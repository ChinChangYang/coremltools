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
NUM_META_CHANNELS = 192  # SGFMetadata::METADATA_INPUT_NUM_CHANNELS

# Metadata channel layout
METADATA_RANK_START_IDX = 6
METADATA_RANK_LEN_PER_PLA = 34
METADATA_DATE_START_IDX = 87
METADATA_DATE_LEN = 32

# Game source constants (matching KataGo SOURCE_* constants)
SOURCE_UNKNOWN = 0
SOURCE_OGS = 1
SOURCE_KGS = 2
SOURCE_FOX = 3
SOURCE_TYGEM = 4
SOURCE_GOGOD = 5
SOURCE_GO4GO = 6


def fill_metadata_row(
    pla_is_human: bool,
    opp_is_human: bool,
    pla_is_unranked: bool,
    opp_is_unranked: bool,
    pla_rank_is_unknown: bool,
    opp_rank_is_unknown: bool,
    inv_pla_rank: int,  # 0=KG, 1=9d, 2=8d, ..., 9=1d, 10=1k, ...
    inv_opp_rank: int,
    game_is_unrated: bool,
    game_ratedness_is_unknown: bool,
    tc_is_unknown: bool,
    tc_is_none: bool,
    tc_is_absolute: bool,
    tc_is_simple: bool,
    tc_is_byoyomi: bool,
    tc_is_canadian: bool,
    tc_is_fischer: bool,
    main_time_seconds: float,
    period_time_seconds: float,
    byoyomi_periods: int,
    canadian_moves: int,
    board_area: int,
    days_since_1970: float,
    source: int,
) -> np.ndarray:
    """Port of SGFMetadata::fillMetadataRow() to Python.

    Generates a 192-channel metadata array for human SL networks.

    Returns:
        np.ndarray of shape (192,) with float32 values.
    """
    import math

    row_metadata = np.zeros(NUM_META_CHANNELS, dtype=np.float32)

    # Channels 0-5: Human/unranked/unknown flags
    row_metadata[0] = 1.0 if pla_is_human else 0.0
    row_metadata[1] = 1.0 if opp_is_human else 0.0
    row_metadata[2] = 1.0 if pla_is_unranked else 0.0
    row_metadata[3] = 1.0 if opp_is_unranked else 0.0
    row_metadata[4] = 1.0 if pla_rank_is_unknown else 0.0
    row_metadata[5] = 1.0 if opp_rank_is_unknown else 0.0

    # Channels 6-73: Rank one-hot encoding (thermometer encoding)
    if not pla_is_unranked:
        for i in range(min(inv_pla_rank, METADATA_RANK_LEN_PER_PLA)):
            row_metadata[METADATA_RANK_START_IDX + i] = 1.0
    if not opp_is_unranked:
        for i in range(min(inv_opp_rank, METADATA_RANK_LEN_PER_PLA)):
            row_metadata[METADATA_RANK_START_IDX + METADATA_RANK_LEN_PER_PLA + i] = 1.0

    # Channel 74: Game rating status
    if game_ratedness_is_unknown:
        row_metadata[74] = 0.5
    elif game_is_unrated:
        row_metadata[74] = 1.0
    else:
        row_metadata[74] = 0.0

    # Channels 75-81: Time control one-hot (exactly one should be 1.0)
    row_metadata[75] = 1.0 if tc_is_unknown else 0.0
    row_metadata[76] = 1.0 if tc_is_none else 0.0
    row_metadata[77] = 1.0 if tc_is_absolute else 0.0
    row_metadata[78] = 1.0 if tc_is_simple else 0.0
    row_metadata[79] = 1.0 if tc_is_byoyomi else 0.0
    row_metadata[80] = 1.0 if tc_is_canadian else 0.0
    row_metadata[81] = 1.0 if tc_is_fischer else 0.0

    # Channels 82-85: Logarithmic time features
    main_time_capped = min(max(main_time_seconds, 0.0), 3.0 * 86400)
    period_time_capped = min(max(period_time_seconds, 0.0), 1.0 * 86400)
    byoyomi_periods_capped = min(max(byoyomi_periods, 0), 50)
    canadian_moves_capped = min(max(canadian_moves, 0), 50)

    row_metadata[82] = 0.4 * (math.log(main_time_capped + 60.0) - 6.5)
    row_metadata[83] = 0.3 * (math.log(period_time_capped + 1.0) - 3.0)
    row_metadata[84] = 0.5 * (math.log(byoyomi_periods_capped + 2.0) - 1.5)
    row_metadata[85] = 0.25 * (math.log(canadian_moves_capped + 2.0) - 1.5)

    # Channel 86: Board area normalization
    row_metadata[86] = 0.5 * math.log(board_area / 361.0)

    # Channels 87-150: Periodic date encoding (32 sin/cos pairs)
    factor = 80000 ** (1.0 / (METADATA_DATE_LEN - 1))
    twopi = 2.0 * math.pi
    period = 7.0  # Start with 7 days (week)

    for i in range(METADATA_DATE_LEN):
        num_revolutions = days_since_1970 / period
        row_metadata[METADATA_DATE_START_IDX + i * 2 + 0] = math.cos(num_revolutions * twopi)
        row_metadata[METADATA_DATE_START_IDX + i * 2 + 1] = math.sin(num_revolutions * twopi)
        period *= factor

    # Channels 151-166: Game source one-hot (16 sources)
    if 0 <= source < 16:
        row_metadata[151 + source] = 1.0

    return row_metadata


def get_basic_rank_profile_metadata(
    inverse_rank: int,
    pre_az: bool,
    board_area: int,
) -> np.ndarray:
    """Generate metadata for a basic rank profile (rank_Xd/Xk).

    Matches makeBasicRankProfile() in KataGo.
    """
    from datetime import date

    # KGS-style settings
    main_time = 1200.0  # 20 minutes
    period_time = 30.0  # 30 seconds
    byoyomi_periods = 5

    # Game date
    if pre_az:
        game_date = date(2016, 9, 1)
    else:
        game_date = date(2020, 3, 1)

    epoch = date(1970, 1, 1)
    days_since_1970 = (game_date - epoch).days

    return fill_metadata_row(
        pla_is_human=True,
        opp_is_human=True,
        pla_is_unranked=False,
        opp_is_unranked=False,
        pla_rank_is_unknown=False,
        opp_rank_is_unknown=False,
        inv_pla_rank=inverse_rank,
        inv_opp_rank=inverse_rank,
        game_is_unrated=False,
        game_ratedness_is_unknown=True,
        tc_is_unknown=False,
        tc_is_none=False,
        tc_is_absolute=False,
        tc_is_simple=False,
        tc_is_byoyomi=True,
        tc_is_canadian=False,
        tc_is_fischer=False,
        main_time_seconds=main_time,
        period_time_seconds=period_time,
        byoyomi_periods=byoyomi_periods,
        canadian_moves=0,
        board_area=board_area,
        days_since_1970=days_since_1970,
        source=SOURCE_KGS,
    )


def get_pro_profile_metadata(
    year: int,
    board_area: int,
) -> np.ndarray:
    """Generate metadata for a pro player profile (proyear_YYYY).

    Matches makeHistoricalProProfile() and makeModernProProfile() in KataGo.
    """
    from datetime import date

    game_date = date(year, 6, 1)
    epoch = date(1970, 1, 1)
    days_since_1970 = (game_date - epoch).days

    # Use GOGOD for historical, GO4GO for modern
    source = SOURCE_GOGOD if year <= 2020 else SOURCE_GO4GO

    return fill_metadata_row(
        pla_is_human=True,
        opp_is_human=True,
        pla_is_unranked=False,
        opp_is_unranked=False,
        pla_rank_is_unknown=False,
        opp_rank_is_unknown=False,
        inv_pla_rank=1,  # 9d
        inv_opp_rank=1,  # 9d
        game_is_unrated=False,
        game_ratedness_is_unknown=False,
        tc_is_unknown=True,
        tc_is_none=False,
        tc_is_absolute=False,
        tc_is_simple=False,
        tc_is_byoyomi=False,
        tc_is_canadian=False,
        tc_is_fischer=False,
        main_time_seconds=0.0,
        period_time_seconds=0.0,
        byoyomi_periods=0,
        canadian_moves=0,
        board_area=board_area,
        days_since_1970=days_since_1970,
        source=source,
    )


def get_profile_metadata(profile_name: str, board_area: int) -> np.ndarray:
    """Generate metadata for a named profile.

    Matches SGFMetadata::getProfile() in KataGo.

    Supported profiles:
    - "rank_9d" through "rank_20k" (post-AlphaZero style)
    - "preaz_9d" through "preaz_20k" (pre-AlphaZero style)
    - "proyear_YYYY" for years 1800-2023

    Args:
        profile_name: Profile name string.
        board_area: Board area (e.g., 361 for 19x19).

    Returns:
        np.ndarray of shape (192,) with float32 values.
    """
    rank_map = {
        "9d": 1, "8d": 2, "7d": 3, "6d": 4, "5d": 5,
        "4d": 6, "3d": 7, "2d": 8, "1d": 9,
        "1k": 10, "2k": 11, "3k": 12, "4k": 13, "5k": 14,
        "6k": 15, "7k": 16, "8k": 17, "9k": 18, "10k": 19,
        "11k": 20, "12k": 21, "13k": 22, "14k": 23, "15k": 24,
        "16k": 25, "17k": 26, "18k": 27, "19k": 28, "20k": 29,
    }

    if profile_name.startswith("rank_"):
        rank_str = profile_name[5:]  # Remove "rank_" prefix
        if rank_str in rank_map:
            return get_basic_rank_profile_metadata(rank_map[rank_str], pre_az=False, board_area=board_area)
    elif profile_name.startswith("preaz_"):
        rank_str = profile_name[6:]  # Remove "preaz_" prefix
        if rank_str in rank_map:
            return get_basic_rank_profile_metadata(rank_map[rank_str], pre_az=True, board_area=board_area)
    elif profile_name.startswith("proyear_"):
        year_str = profile_name[8:]  # Remove "proyear_" prefix
        try:
            year = int(year_str)
            if 1800 <= year <= 2023:
                return get_pro_profile_metadata(year, board_area)
        except ValueError:
            pass

    raise ValueError(f"Unknown profile: {profile_name}")


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


# ============================================================================
# Metadata test generators for human SL networks
# ============================================================================


def generate_zeros_with_metadata_test(board_x_size: int, board_y_size: int, profile: str = "rank_1d") -> dict:
    """Generate empty board with metadata test case for human SL networks."""
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    spatial[0, :, :] = 1.0  # Valid board mask

    board_area = board_x_size * board_y_size
    meta_input = get_profile_metadata(profile, board_area)

    return {
        "name": f"zeros_with_metadata_{profile}",
        "description": f"Empty board ({board_x_size}x{board_y_size}) with {profile} metadata",
        "spatial_input": spatial.tolist(),
        "global_input": np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32).tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
        "meta_input": meta_input.tolist(),
    }


def generate_rank_1d_metadata_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate test with 1 dan player metadata."""
    return generate_zeros_with_metadata_test(board_x_size, board_y_size, "rank_1d")


def generate_rank_5k_metadata_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate test with 5 kyu player metadata."""
    return generate_zeros_with_metadata_test(board_x_size, board_y_size, "rank_5k")


def generate_rank_10k_metadata_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate test with 10 kyu player metadata."""
    return generate_zeros_with_metadata_test(board_x_size, board_y_size, "rank_10k")


def generate_proyear_2020_metadata_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate test with professional player 2020 metadata."""
    return generate_zeros_with_metadata_test(board_x_size, board_y_size, "proyear_2020")


def generate_random_with_metadata_test(board_x_size: int, board_y_size: int) -> dict:
    """Generate random board with 1d metadata test case."""
    np.random.seed(42)
    spatial = np.zeros((NUM_SPATIAL_CHANNELS, board_y_size, board_x_size), dtype=np.float32)
    spatial[0, :, :] = 1.0  # Valid board mask

    # Randomly place stones
    for channel in range(1, 7):
        random_mask = np.random.rand(board_y_size, board_x_size) > 0.7
        spatial[channel, random_mask] = 1.0

    global_in = np.zeros(NUM_GLOBAL_CHANNELS, dtype=np.float32)
    global_in[0] = 1.0  # Side to move
    global_in[5] = 0.375  # Normalized komi

    board_area = board_x_size * board_y_size
    meta_input = get_profile_metadata("rank_1d", board_area)

    return {
        "name": "random_with_metadata",
        "description": f"Random board ({board_x_size}x{board_y_size}) with rank_1d metadata",
        "spatial_input": spatial.tolist(),
        "global_input": global_in.tolist(),
        "input_mask": np.ones((board_y_size, board_x_size), dtype=np.float32).tolist(),
        "meta_input": meta_input.tolist(),
    }


def save_test_case(test_case: dict, output_dir: Path, board_x_size: int, board_y_size: int) -> None:
    """Save a test case to JSON file.

    Args:
        test_case: Dictionary with test case data
        output_dir: Output directory
        board_x_size: Board width
        board_y_size: Board height
    """
    name = test_case["name"]
    filepath = output_dir / f"{name}.json"

    # Add board dimensions to test case
    test_case_with_dims = {
        "boardXSize": board_x_size,
        "boardYSize": board_y_size,
        **test_case
    }

    with open(filepath, "w") as f:
        json.dump(test_case_with_dims, f, indent=2)

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
    parser.add_argument(
        "--with-metadata",
        action="store_true",
        help="Generate additional test cases with metadata for human SL networks"
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

    # Add metadata test generators if requested
    if args.with_metadata:
        metadata_generators = [
            generate_rank_1d_metadata_test,
            generate_rank_5k_metadata_test,
            generate_rank_10k_metadata_test,
            generate_proyear_2020_metadata_test,
            generate_random_with_metadata_test,
        ]
        test_generators.extend(metadata_generators)

    for generator in test_generators:
        test_case = generator(board_x_size, board_y_size)
        save_test_case(test_case, output_dir, board_x_size, board_y_size)

    print(f"\nGenerated {len(test_generators)} test cases")

    # Also create an index file
    index = {
        "description": "KataGo to Core ML cross-validation test inputs",
        "board_x_size": board_x_size,
        "board_y_size": board_y_size,
        "num_spatial_channels": NUM_SPATIAL_CHANNELS,
        "num_global_channels": NUM_GLOBAL_CHANNELS,
        "num_meta_channels": NUM_META_CHANNELS if args.with_metadata else 0,
        "test_cases": [gen(board_x_size, board_y_size)["name"] for gen in test_generators]
    }

    index_path = output_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"  Created index: {index_path}")


if __name__ == "__main__":
    main()
