# KataGo to Core ML Converter - API Reference

Python API documentation for converting KataGo models to Core ML format.

## Installation

This converter is part of the KataGoCoremltools fork of coremltools:

```bash
git clone https://github.com/ChinChangYang/coremltools.git
cd coremltools
git checkout katagocoremltools
make build
source scripts/env_activate.sh --python=3.11
```

## Quick Start

```python
import coremltools as ct

# Convert KataGo model to Core ML
mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS15
)

# Save the model
mlmodel.save("KataGo.mlpackage")

# Run inference
import numpy as np
result = mlmodel.predict({
    "spatial_input": np.random.randn(1, 22, 19, 19).astype(np.float32),
    "global_input": np.random.randn(1, 19).astype(np.float32),
    "input_mask": np.ones((1, 1, 19, 19), dtype=np.float32)
})
```

## API Reference

### `convert()`

Convert a KataGo binary model to Core ML format.

**Signature:**
```python
def convert(
    model_path: str,
    minimum_deployment_target: Optional[ct.target] = None
) -> ct.models.MLModel
```

**Parameters:**

- **model_path** (`str`, required):
  - Path to KataGo model file
  - Supports `.bin` or `.bin.gz` formats
  - Must be KataGo model version 15 or 16
  - Example: `"kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"`

- **minimum_deployment_target** (`ct.target`, optional):
  - Minimum OS version for deployment
  - Options:
    - `ct.target.iOS15` (recommended)
    - `ct.target.iOS16`
    - `ct.target.iOS17`
    - `ct.target.macOS12`
    - `ct.target.macOS13`
  - Default: Latest iOS version supported by coremltools
  - Affects available operations and optimizations

**Returns:**

- `ct.models.MLModel`: Core ML model ready for prediction or saving

**Raises:**

- `ValueError`: If model version is not 15 or 16
- `FileNotFoundError`: If model_path doesn't exist
- `RuntimeError`: If conversion fails

**Example:**
```python
import coremltools as ct

# Basic conversion
mlmodel = ct.converters.katago.convert("model.bin.gz")

# With specific deployment target
mlmodel = ct.converters.katago.convert(
    "model.bin.gz",
    minimum_deployment_target=ct.target.iOS15
)

# Save and use
mlmodel.save("KataGo.mlpackage")
```

## Model Inputs

The converted Core ML model expects three inputs:

### `spatial_input`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 22, 19, 19)`
- **Description**: Spatial features of the board state
- **Channels** (22 total):
  1. Current player stones (19×19 grid)
  2. Opponent player stones (19×19 grid)
  3-22. Historical board positions and features

**Example:**
```python
import numpy as np
spatial_input = np.zeros((1, 22, 19, 19), dtype=np.float32)

# Set current player stone at (3, 3)
spatial_input[0, 0, 3, 3] = 1.0

# Set opponent stone at (15, 15)
spatial_input[0, 1, 15, 15] = 1.0
```

### `global_input`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 19)`
- **Description**: Global game state features
- **Features** (19 total):
  1. Komi value (normalized)
  2. Game rules encoding
  3. Ko rule variant
  4. Scoring rules
  5. Tax rules
  6. Suicide allowed flag
  7. Multi-stone suicide allowed flag
  8. Button Go mode
  9. White handicap bonus
  10-19. Additional global features

**Example:**
```python
global_input = np.zeros((1, 19), dtype=np.float32)
global_input[0, 0] = 7.5 / 20.0  # Komi = 7.5, normalized by 20
```

### `input_mask`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 1, 19, 19)`
- **Description**: Mask indicating valid board positions
- **Values**:
  - `1.0` = valid position for moves
  - `0.0` = invalid position (e.g., already occupied)

**Example:**
```python
input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)

# Mark position (3, 3) as invalid (occupied)
input_mask[0, 0, 3, 3] = 0.0
```

## Model Outputs

The converted model produces five outputs:

### `policy`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 2, 19, 19)`
- **Description**: Move policy logits for each board position
- **Channels**:
  - Channel 0: Regular move policy
  - Channel 1: Alternative policy head (if present)

**Usage:**
```python
policy = result['policy']
# Get policy for position (10, 10)
move_logit = policy[0, 0, 10, 10]
```

### `pass_policy`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 1)`
- **Description**: Policy logit for passing

**Usage:**
```python
pass_logit = result['pass_policy'][0, 0]
```

### `value`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 3)`
- **Description**: Game outcome probabilities
- **Elements**:
  - Index 0: Win probability for current player
  - Index 1: Loss probability for current player
  - Index 2: Draw probability

**Usage:**
```python
win_prob = result['value'][0, 0]
loss_prob = result['value'][0, 1]
draw_prob = result['value'][0, 2]
```

### `ownership`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 1, 19, 19)`
- **Description**: Predicted final ownership of each point
- **Values**:
  - Positive = current player's territory
  - Negative = opponent's territory
  - Zero = neutral

**Usage:**
```python
ownership = result['ownership'][0, 0]
# Get ownership prediction for position (5, 5)
point_ownership = ownership[5, 5]
```

### `score_value`

- **Type**: Float32 MultiArray
- **Shape**: `(1, 6)`
- **Description**: Score distribution features
- **Elements**: Score mean, variance, and distribution parameters

**Usage:**
```python
score_mean = result['score_value'][0, 0]
score_variance = result['score_value'][0, 1]
```

## Complete Example

```python
#!/usr/bin/env python3
"""Complete example of KataGo to Core ML conversion and inference."""

import numpy as np
import coremltools as ct

# 1. Convert model
print("Converting KataGo model...")
mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS15
)

# 2. Save model
output_path = "KataGo.mlpackage"
mlmodel.save(output_path)
print(f"Model saved to {output_path}")

# 3. Reload and test
print("\nTesting inference...")
model = ct.models.MLModel(output_path)

# 4. Create sample inputs
batch_size = 1
board_size = 19
spatial_channels = 22
global_features = 19

inputs = {
    "spatial_input": np.random.randn(
        batch_size, spatial_channels, board_size, board_size
    ).astype(np.float32),
    "global_input": np.random.randn(
        batch_size, global_features
    ).astype(np.float32),
    "input_mask": np.ones(
        (batch_size, 1, board_size, board_size),
        dtype=np.float32
    )
}

# 5. Run inference
result = model.predict(inputs)

# 6. Process outputs
print("\nOutput shapes:")
for key, value in result.items():
    print(f"  {key}: {value.shape}")

# 7. Extract specific predictions
policy = result['policy'][0, 0]  # First channel, first batch
best_move_idx = np.unravel_index(np.argmax(policy), policy.shape)
print(f"\nBest move (row, col): {best_move_idx}")

value = result['value'][0]
print(f"Win probability: {value[0]:.3f}")
print(f"Loss probability: {value[1]:.3f}")
print(f"Draw probability: {value[2]:.3f}")

ownership = result['ownership'][0, 0]
print(f"\nOwnership range: [{ownership.min():.3f}, {ownership.max():.3f}]")

print("\nInference test completed successfully!")
```

## Model Format Details

### Supported KataGo Versions

- **Version 15**: Fully supported
- **Version 16**: Fully supported
- **Earlier versions**: Not supported (will raise `ValueError`)

### Supported Architectures

- **Residual blocks**: Standard ResNet-style blocks
- **Bottleneck blocks**: Nested bottleneck blocks with dilated convolutions
- **Global pooling blocks**: Mean/max pooling across spatial dimensions
- **Activations**: ReLU, Mish (identity for pass-through)

### Board Size

- **Fixed**: 19×19 only
- Other board sizes (9×9, 13×13) are not supported in the current version

### Input Compression

- **Gzip compressed** (`.bin.gz`): Automatically decompressed
- **Uncompressed** (`.bin`): Directly parsed

## Performance Considerations

### Inference Speed

- **Apple Silicon (M1/M2/M3)**: 5-20ms per inference
  - Neural Engine acceleration
  - Optimized for on-device inference

- **Intel Mac**: 50-200ms per inference
  - CPU/GPU only
  - Slower but functional

### Memory Usage

- **Model size**: ~250-300MB (similar to input .bin.gz)
- **Runtime memory**: ~500MB-1GB depending on model size
- **Peak memory during conversion**: ~2-3GB

### Optimization Tips

1. **Use appropriate compute units**:
   ```python
   model = ct.models.MLModel(
       "KataGo.mlpackage",
       compute_units=ct.ComputeUnit.ALL  # Use Neural Engine + GPU
   )
   ```

2. **Reuse model instance**:
   ```python
   # Good: Load once
   model = ct.models.MLModel("KataGo.mlpackage")
   for position in positions:
       result = model.predict(position)

   # Bad: Load repeatedly
   for position in positions:
       model = ct.models.MLModel("KataGo.mlpackage")
       result = model.predict(position)
   ```

3. **Minimize data copies**:
   ```python
   # Reuse input arrays when possible
   spatial_input = np.zeros((1, 22, 19, 19), dtype=np.float32)
   for position in positions:
       # Update spatial_input in-place
       spatial_input[...] = position
       result = model.predict({"spatial_input": spatial_input, ...})
   ```

## Troubleshooting

For detailed troubleshooting, see [docs/katago/TROUBLESHOOTING.md](../../docs/katago/TROUBLESHOOTING.md).

Common issues:

- **"Only KataGo model versions (15, 16) are supported"**: Download a newer model
- **Shape mismatch errors**: Verify input array shapes match specification
- **Slow inference**: Ensure running on Apple Silicon, use `ComputeUnit.ALL`
- **Import errors**: Verify environment activation with `source scripts/env_activate.sh`

## See Also

- [Main Documentation](../../docs/katago/README.md) - Complete setup guide
- [Quick Start](../../docs/katago/QUICK_START.md) - Quick reference
- [Troubleshooting](../../docs/katago/TROUBLESHOOTING.md) - Error solutions
- [Backend Integration](../../docs/katago/BACKEND_INTEGRATION.md) - KataGo integration guide

## License

This converter is part of coremltools and follows the same license terms.

---

**Last updated**: 2025-12-19
