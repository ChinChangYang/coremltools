# KataGo Core ML Test Inputs

This directory contains test inputs for cross-validating KataGo Core ML conversions against the KataGo Eigen backend.

## Directory Structure

```
test_inputs/
├── 9x9/          # Test inputs for 9x9 board
├── 13x13/        # Test inputs for 13x13 board
├── 19x19/        # Test inputs for 19x19 board
└── README.md     # This file
```

Each board size directory contains 9 test cases in JSON format, plus an `index.json` metadata file.

## Test Cases

Each directory contains the following test cases:

1. **zeros.json** - Empty board (all inputs zero)
   - Simplest baseline test case
   - Tests model behavior with no features

2. **random_seed_42.json** - Reproducible random binary patterns
   - Uses fixed random seed for reproducibility
   - Tests model with varied input patterns

3. **corner_stone.json** - Single stone at corner position (0,0)
   - Tests corner handling
   - Minimal non-zero input

4. **center_stone.json** - Single stone at board center
   - Tests center position handling
   - Symmetry validation

5. **partial_mask_NxN.json** - Half board masked (upper-left quadrant valid)
   - Test case name varies by board size:
     - 9x9: `partial_mask_4x4`
     - 13x13: `partial_mask_6x6`
     - 19x19: `partial_mask_9x9`
   - Tests mask handling and edge cases

6. **edge_pattern.json** - Alternating stones along top and left edges
   - Tests edge position handling
   - More complex pattern than single stone

7. **diagonal_pattern.json** - Diagonal line of stones
   - Tests diagonal symmetry
   - Pattern across the board

8. **uniform_small.json** - Complex multi-stone pattern
   - Multiple stones in upper-left corner
   - Tests feature interaction

9. **komi_7_5.json** - Empty board with komi=7.5
   - Tests global input handling (komi value)
   - Empty spatial input

## Generating Test Inputs

To regenerate test inputs for all board sizes:

```bash
# Generate 9x9 test inputs
python scripts/generate_test_inputs.py --output test_inputs/9x9 --board-x-size 9 --board-y-size 9

# Generate 13x13 test inputs
python scripts/generate_test_inputs.py --output test_inputs/13x13 --board-x-size 13 --board-y-size 13

# Generate 19x19 test inputs
python scripts/generate_test_inputs.py --output test_inputs/19x19 --board-x-size 19 --board-y-size 19
```

## Test Input Format

Each test case is a JSON file with the following structure:

```json
{
  "name": "test_case_name",
  "description": "Human-readable description",
  "spatial_input": [22][H][W],  // 22 spatial channels (KataGo v7)
  "global_input": [19],          // 19 global channels
  "input_mask": [H][W]           // Boolean board mask (0.0 or 1.0)
}
```

Where:
- `H` = board height (board_y_size)
- `W` = board width (board_x_size)

### Spatial Input Channels (22 channels)

- Channel 0: Valid board mask (1.0 for valid positions)
- Channels 1-6: Stone positions and board liberties (binary/0-1 range)
- Channels 7-22: Various position features

### Global Input Channels (19 channels)

- Side to move indicator
- Komi value (normalized)
- Move history features
- Board state summary statistics

### Input Mask

- Float32 values (0.0 or 1.0)
- Indicates valid board positions
- Can be partial for testing edge cases

## Using Test Inputs

### Cross-Validation Script

```bash
python scripts/validate_coreml.py \
    --model-mlpackage KataGo-9x9.mlpackage \
    --model-bin katago_eigen/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz \
    --katago-exe katago_eigen/cpp/build/katago \
    --test-inputs test_inputs/9x9
```

### Pytest Tests

```bash
# Run all cross-validation tests (all board sizes)
pytest coremltools/test/converters/katago/ -v

# Run tests for specific board size
pytest coremltools/test/converters/katago/ -v -k "9"

# Run specific test case across all board sizes
pytest coremltools/test/converters/katago/ -v -k "zeros"
```

## Metadata (index.json)

Each directory contains an `index.json` file with metadata:

```json
{
  "description": "KataGo to Core ML cross-validation test inputs",
  "board_x_size": 9,
  "board_y_size": 9,
  "num_spatial_channels": 22,
  "num_global_channels": 19,
  "test_cases": [
    "zeros",
    "random_seed_42",
    ...
  ]
}
```

## Notes

- Test inputs are generated deterministically using fixed random seeds where applicable
- All test cases use the same KataGo model version (v7 with 22 spatial channels)
- Tolerance thresholds for cross-validation are defined in `validation_utils.py`
- Test inputs can be regenerated at any time without affecting test results (deterministic)
