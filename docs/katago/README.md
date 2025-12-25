# KataGo to Core ML Converter

Complete guide for converting KataGo neural network models to Core ML format on macOS.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites Installation](#prerequisites-installation)
3. [Repository Setup](#repository-setup)
4. [Build coremltools from Source](#build-coremltools-from-source)
5. [Download a KataGo Model](#download-a-katago-model)
6. [Convert KataGo Model to Core ML](#convert-katago-model-to-core-ml)
7. [Test the Converted Model](#test-the-converted-model)
8. [Using the Model](#using-the-model)
9. [Troubleshooting](#troubleshooting)
10. [Future Work: KataGo Core ML Backend Integration](#future-work-katago-core-ml-backend-integration)
11. [Verification Checklist](#verification-checklist)

## Introduction

### What is this?

This guide walks you through converting KataGo neural network models from their binary format (.bin, .bin.gz) to Apple's Core ML format (.mlpackage). Core ML models can run efficiently on Mac and iOS devices, leveraging Apple Silicon's Neural Engine for optimal performance.

**KataGo** is a strong open-source Go AI that uses deep neural networks for position evaluation and move selection. By converting KataGo models to Core ML, you can:
- Run inference on Apple devices without external dependencies
- Leverage Apple Silicon's Neural Engine for efficient computation
- Integrate Go AI into macOS and iOS applications
- Experiment with on-device AI for the game of Go

### Requirements

- **macOS**: 10.15 (Catalina) or later, preferably macOS 12+ on Apple Silicon
- **Disk Space**: ~10GB free (for build dependencies and model files)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Internet**: Broadband connection for downloading dependencies and models
- **Time**: 60-90 minutes for first-time setup (mostly automated building)

### What You'll Get

After following this guide, you will have:
- A functional coremltools installation with KataGo converter support
- A converted KataGo model in `.mlpackage` format
- Verified inference capability on your Mac
- Understanding of how to integrate the model into applications

The converted model will accept:
- **Input**: Board position (22 feature planes), global state (19 features), position mask
- **Output**: Policy probabilities, value estimates, ownership predictions, score distribution

**Supported Models:**
- KataGo model versions 15 and 16
- Board size: 19x19 only
- Model architectures: All KataGo network types (ordinary blocks, global pooling blocks, nested bottleneck blocks)

---

## Prerequisites Installation

### 1. Install Xcode Command Line Tools

Xcode Command Line Tools provide essential development utilities including compilers, git, and build tools.

```bash
xcode-select --install
```

This will open a dialog asking you to install the tools. Click **Install** and agree to the license. The download is ~500MB and will take 5-10 minutes.

**Verify installation:**

```bash
xcode-select -p
# Should output: /Library/Developer/CommandLineTools or /Applications/Xcode.app/Contents/Developer

gcc --version
# Should output: Apple clang version 14.0.0 or later
```

If you see "command line tools are already installed", you're all set.

### 2. Install Miniconda

Miniconda is a minimal Python distribution manager that will create an isolated environment for building coremltools.

**For Apple Silicon (M1/M2/M3):**

```bash
# Download the installer
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install (this will install to ~/miniconda3 by default)
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# Initialize shell integration
~/miniconda3/bin/conda init zsh

# Restart terminal or source configuration
source ~/.zshrc

# Verify installation
conda --version
# Should output: conda 24.x.x or later
```

**For Intel Mac:**

Replace `arm64` with `x86_64` in the download URL:
```bash
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

### 3. Verify Git Installation

Git should be installed with Xcode Command Line Tools:

```bash
git --version
# Should output: git version 2.x.x or later
```

If not installed, run `xcode-select --install` again.

---

## Repository Setup

### 1. Clone coremltools with KataGo Support

```bash
# Create a workspace directory
mkdir -p ~/katago_workspace
cd ~/katago_workspace

# Clone the coremltools fork with KataGo converter
git clone https://github.com/ChinChangYang/coremltools.git
cd coremltools

# Switch to katagocoremltools branch
git checkout katagocoremltools
```

**Expected output:**
```
Cloning into 'coremltools'...
remote: Enumerating objects: 123456, done.
remote: Counting objects: 100% (12345/12345), done.
remote: Compressing objects: 100% (5678/5678), done.
remote: Total 123456 (delta 67890), reused 111222 (delta 55666), pack-reused 111111
Receiving objects: 100% (123456/123456), 45.67 MiB | 10.23 MiB/s, done.
Resolving deltas: 100% (67890/67890), done.

Branch 'katagocoremltools' set up to track remote branch 'katagocoremltools' from 'origin'.
Switched to a new branch 'katagocoremltools'
```

**Verify KataGo converter exists:**

```bash
ls coremltools/converters/katago/
```

**Expected output:**
```
__init__.py
_converter.py
_katago_model_builder.py
_katago_ops.py
_katago_parser.py
_katago_types.py
```

If you see these files, the KataGo converter is present.

### 2. Verify Configuration

The repository should already have the correct build settings for macOS. Verify:

**Check Makefile (Python version):**
```bash
grep "^python = " Makefile
```
Expected output: `python = 3.11`

**Check CMakeLists.txt (architecture configuration):**
```bash
grep "PLAT_NAME" CMakeLists.txt | grep CMAKE_OSX_ARCHITECTURES
```
Should show configuration using `CMAKE_OSX_ARCHITECTURES`.

**Check scripts/build.sh (ARM64 architecture):**
```bash
grep "CMAKE_OSX_ARCHITECTURES" scripts/build.sh
```
Should show: `-DCMAKE_OSX_ARCHITECTURES=arm64`

**Note for Intel Macs:** If you're on an Intel Mac, you'll need to change `arm64` to `x86_64` in `scripts/build.sh` line ~115.

---

## Build coremltools from Source

Building coremltools from source compiles the C++ extensions and creates a Python package with the KataGo converter.

### 1. Create Conda Environment and Build

**⏱️ Expected time: 30-60 minutes** (mostly automated, depends on CPU and internet speed)

```bash
# From the coremltools directory
make build

# If build fails with "ld: library 'c++' not found", use:
# CXX=/usr/bin/clang++ CC=/usr/bin/clang make build
```

**What this does:**
1. Creates a conda environment at `envs/KataGoCoremltools-py3.11`
2. Installs Python 3.11 and all dependencies (numpy, protobuf, etc.)
3. Builds C++ extensions:
   - `libmilstoragepython` - MIL storage layer
   - `libcoremlpython` - Core ML Python bindings
   - `libmodelpackage` - Model package utilities
4. Builds kmeans1d dependency
5. Copies necessary files to build directory

**Progress indicators:**

You'll see output like:
```
Creating a new conda environment in .../envs/KataGoCoremltools-py3.11
Collecting package metadata (current_repodata.json): done
Solving environment: done
Installing pip dependencies: ...
Configuring with CMake...
-- The CXX compiler identification is AppleClang 14.0.0
-- Detecting CXX compiler ABI info
-- Configuring done
-- Generating done
-- Build files written to: .../build
[ 10%] Building CXX object deps/protobuf/src/CMakeFiles/libprotobuf.dir/...
[ 20%] Building CXX object mlmodel/src/CMakeFiles/mlmodel.dir/...
...
[100%] Built target coremlpython
```

**Common build messages (normal):**
- "Warning: Unused variable" - These are harmless warnings from dependencies
- "Note: including file: ..." - Normal include file processing
- CMake policy warnings - Can be ignored

**Troubleshooting:**

**Problem:** `conda: command not found`
```bash
# Restart your terminal
source ~/.zshrc
conda --version
```

**Problem:** CMake errors about compiler
```bash
# Ensure Xcode Command Line Tools are installed
xcode-select --install
```

**Problem:** "Could not find numpy include path"
```bash
# The build script should handle this, but if it fails:
conda install numpy
```

### 2. Activate the Build Environment

After the build completes, activate the conda environment:

```bash
source scripts/env_activate.sh --python=3.11
```

This script sets up environment variables:
- `PYTHON_EXECUTABLE` - Path to Python 3.11
- `PYTHON_INCLUDE_DIR` - Python header files
- `PYTHON_LIBRARY` - Python library files
- `PATH` - Updated to use conda environment

**Verify activation:**

```bash
which python
# Should output: .../KataGoCoremltools/envs/KataGoCoremltools-py3.11/bin/python

python --version
# Should output: Python 3.11.x
```

### 3. Verify coremltools Installation

```bash
python -c "import coremltools as ct; print(ct.__version__)"
```

**Expected output:** `9.0`

### 4. Verify KataGo Converter

```bash
python -c "from coremltools.converters import katago; print('KataGo converter available')"
```

**Expected output:** `KataGo converter available`

If you see this message, the KataGo converter is successfully installed!

---

## Download a KataGo Model

### 1. Download Test Model

We'll use a production-quality KataGo model for testing:

```bash
# Return to workspace directory
cd ~/katago_workspace

# Download the model (~271MB compressed)
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz

# Verify download
ls -lh kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

**Expected output:**
```
-rw-r--r--  1 user  staff   271M Dec 19 10:00 kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

**Download time:** ~1-3 minutes on broadband connection

### 2. Model Information

This model (`kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz`):
- **Version**: 15 (fully supported by converter)
- **Architecture**: 28 nested bottleneck blocks, 512 channels
- **Training**: 11.165 billion positions, 5.387 billion data
- **Input channels**: 22 spatial + 19 global
- **Board size**: 19x19 (only size supported by converter)
- **Strength**: Professional level (estimated 9-10 dan)

### 3. Alternative Models

For other models, visit:
- **Official KataGo networks**: https://katagotraining.org/networks/
- **Kata1 models**: Look for `kata1-b*` files
- **Requirements**: Must be version 15 or 16

**Model size guide:**
- Small (testing): 128-256 channels, 10-20 blocks (~50-100MB)
- Medium: 384-512 channels, 20-30 blocks (~150-300MB)
- Large (strongest): 512+ channels, 30-40 blocks (~400-800MB)

---

## Convert KataGo Model to Core ML

### 1. Create Conversion Script

Create a Python script to convert the model:

```bash
cd ~/katago_workspace

cat > convert_katago.py << 'EOF'
#!/usr/bin/env python3
"""Convert KataGo model to Core ML format."""

import sys
import coremltools as ct

def main():
    model_path = "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz"
    output_path = "KataGo.mlpackage"

    print(f"Converting {model_path}...")
    print("This may take 5-10 minutes...")

    try:
        # Convert the model
        mlmodel = ct.converters.katago.convert(
            model_path,
            minimum_deployment_target=ct.target.iOS15
        )

        print(f"\nConversion successful!")
        print(f"Saving to {output_path}...")

        # Save the model
        mlmodel.save(output_path)

        print(f"\nModel saved successfully!")
        print(f"\nModel Information:")
        print(f"  Inputs:")
        for inp in mlmodel.input_description:
            print(f"    - {inp.name}: {inp.type}")
        print(f"  Outputs:")
        for out in mlmodel.output_description:
            print(f"    - {out.name}: {out.type}")

        return 0

    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x convert_katago.py
```

### 2. Run Conversion

**⏱️ Expected time: 5-10 minutes**

```bash
# Ensure environment is activated
source ~/katago_workspace/coremltools/scripts/env_activate.sh --python=3.11

# Run conversion
python convert_katago.py
```

**Expected output:**
```
Converting kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz...
This may take 5-10 minutes...

Parsing model...
Building MIL program...
Converting to Core ML...

Conversion successful!
Saving to KataGo.mlpackage...

Model saved successfully!

Model Information:
  Inputs:
    - spatial_input: multiarray (Float32 1 × 22 × 19 × 19)
    - global_input: multiarray (Float32 1 × 19)
    - input_mask: multiarray (Float32 1 × 1 × 19 × 19)
  Outputs:
    - policy: multiarray (Float32 1 × 2 × 19 × 19)
    - pass_policy: multiarray (Float32 1 × 1)
    - value: multiarray (Float32 1 × 3)
    - ownership: multiarray (Float32 1 × 1 × 19 × 19)
    - score_value: multiarray (Float32 1 × 6)
```

**What's happening:**
1. **Parsing** (~30 seconds): Reads binary model file, decompresses, parses layers
2. **Building MIL program** (~2-3 minutes): Constructs intermediate representation
3. **Converting to Core ML** (~2-4 minutes): Optimizes and generates .mlpackage

### 3. Verify Output

```bash
# Check the output package structure
ls -lh KataGo.mlpackage/

# Check total size
du -sh KataGo.mlpackage/
```

**Expected output:**
```
KataGo.mlpackage/
├── Data/
│   └── com.apple.CoreML/
│       ├── model.mlmodel
│       └── weights/
│           └── weight.bin
└── Manifest.json

Total size: ~250-300MB
```

The .mlpackage is a bundle containing:
- **Manifest.json**: Package metadata
- **model.mlmodel**: Model architecture and metadata
- **weights/weight.bin**: Neural network weights

---

## Test the Converted Model

### 1. Create Test Script

Create a script to test inference:

```bash
cd ~/katago_workspace

cat > test_inference.py << 'EOF'
#!/usr/bin/env python3
"""Test inference with the converted Core ML model."""

import numpy as np
import coremltools as ct

def main():
    model_path = "KataGo.mlpackage"

    print(f"Loading {model_path}...")
    mlmodel = ct.models.MLModel(model_path)

    print("Creating random input...")
    # Create random inputs (not a real board position)
    inputs = {
        "spatial_input": np.random.randn(1, 22, 19, 19).astype(np.float32),
        "global_input": np.random.randn(1, 19).astype(np.float32),
        "input_mask": np.ones((1, 1, 19, 19), dtype=np.float32)
    }

    print("Running inference...")
    outputs = mlmodel.predict(inputs)

    print("\nInference Results:")
    for key, value in outputs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Min: {value.min():.4f}, Max: {value.max():.4f}")
            print(f"    Mean: {value.mean():.4f}")
        else:
            print(f"  {key}: {type(value)}")

    print("\nInference test PASSED!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
EOF

chmod +x test_inference.py
```

### 2. Run Test

**⏱️ Expected time: 5-10 seconds (includes model loading)**

```bash
python test_inference.py
```

**Expected output:**
```
Loading KataGo.mlpackage...
Creating random input...
Running inference...

Inference Results:
  policy:
    Shape: (1, 2, 19, 19)
    Min: -18.2341, Max: 15.4567
    Mean: 0.1234
  pass_policy:
    Shape: (1, 1)
    Min: -3.4567, Max: -3.4567
    Mean: -3.4567
  value:
    Shape: (1, 3)
    Min: -2.1234, Max: 1.8765
    Mean: 0.2345
  ownership:
    Shape: (1, 1, 19, 19)
    Min: -0.9876, Max: 0.9234
    Mean: 0.0123
  score_value:
    Shape: (1, 6)
    Min: -4.5678, Max: 2.3456
    Mean: -0.7890

Inference test PASSED!
```

**Note:** The actual values will vary because we're using random inputs. The important thing is that:
- All output shapes are correct
- No errors or exceptions occur
- Inference completes in reasonable time (<1 second after model load)

**Inference performance:**
- First inference (cold): ~500ms (includes model compilation)
- Subsequent inferences: ~50-100ms on Apple Silicon (Neural Engine)

---

## Using the Model

### 1. Integration Options

#### Swift (iOS/macOS Applications)

```swift
import CoreML
import CoreImage

// Load the model
guard let model = try? KataGo(configuration: MLModelConfiguration()) else {
    fatalError("Failed to load KataGo model")
}

// Prepare inputs
let spatialInput = try! MLMultiArray(shape: [1, 22, 19, 19], dataType: .float32)
let globalInput = try! MLMultiArray(shape: [1, 19], dataType: .float32)
let inputMask = try! MLMultiArray(shape: [1, 1, 19, 19], dataType: .float32)

// TODO: Fill inputs with actual board features

// Run inference
let input = KataGoInput(
    spatial_input: spatialInput,
    global_input: globalInput,
    input_mask: inputMask
)

guard let output = try? model.prediction(input: input) else {
    fatalError("Inference failed")
}

// Access outputs
let policy = output.policy  // MLMultiArray (1, 2, 19, 19)
let value = output.value    // MLMultiArray (1, 3)
let ownership = output.ownership  // MLMultiArray (1, 1, 19, 19)
```

#### Python (Testing and Development)

```python
import coremltools as ct
import numpy as np

# Load model
model = ct.models.MLModel("KataGo.mlpackage")

# Prepare inputs
# TODO: Generate actual board features from game position
spatial_input = np.zeros((1, 22, 19, 19), dtype=np.float32)
global_input = np.zeros((1, 19), dtype=np.float32)
input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)

# Run inference
result = model.predict({
    "spatial_input": spatial_input,
    "global_input": global_input,
    "input_mask": input_mask
})

# Access outputs
policy = result["policy"]      # shape: (1, 2, 19, 19)
value = result["value"]        # shape: (1, 3)
ownership = result["ownership"]  # shape: (1, 1, 19, 19)
```

### 2. Input Format Details

#### `spatial_input` (1, 22, 19, 19) - Float32

22 feature planes representing the board state in NCHW format:

**Planes 0-6**: Current player's stones (past 7 moves)
- Plane 0: Current player's stones from this turn
- Plane 1: Current player's stones from 1 turn ago
- ...
- Plane 6: Current player's stones from 6 turns ago

**Planes 7-13**: Opponent's stones (past 7 moves)
- Plane 7: Opponent's stones from this turn
- ...
- Plane 13: Opponent's stones from 6 turns ago

**Plane 14**: Ko-prohibited locations
**Plane 15**: Pass-alive territory
**Plane 16-21**: Additional board features (ladders, captures, etc.)

All values are binary (0.0 or 1.0) except for some features which use float values.

#### `global_input` (1, 19) - Float32

19 global features describing the game state:
- Komi (points given to white)
- Game rules (Chinese, Japanese, etc.)
- Ko rule variations
- Scoring method
- Tax rules
- Suicide legality
- Multi-stone suicide
- Pass behavior
- Board size indicator
- And other game-state features

Values are typically in range [-1.0, 1.0] after normalization.

#### `input_mask` (1, 1, 19, 19) - Float32

Binary mask indicating valid board positions:
- **1.0**: Valid position for this board size
- **0.0**: Invalid/outside board

For 19x19 boards, all values should be 1.0. This input exists to support future variable board sizes.

### 3. Output Format Details

#### `policy` (1, 2, 19, 19) - Float32

Move policy logits for each board position:
- **Channel 0**: Base move policy
- **Channel 1**: Policy optimism (for uncertainty-aware search)

**Usage**: Apply softmax to get move probabilities:
```python
policy_logits = result["policy"][0, 0]  # Use channel 0
policy_probs = scipy.special.softmax(policy_logits.flatten())
policy_probs = policy_probs.reshape(19, 19)
```

Higher values indicate more likely moves. Combine with `pass_policy` for the full policy distribution.

#### `pass_policy` (1, 1) - Float32

Logit for the "pass" move. Single scalar value.

**Usage**: Combine with spatial policy:
```python
pass_logit = result["pass_policy"][0, 0]
# Include in softmax calculation with spatial policy
```

#### `value` (1, 3) - Float32

Game outcome predictions:
- **Index 0**: Win probability for current player
- **Index 1**: Loss probability for current player
- **Index 2**: No-result probability (jigo, draw)

Values are logits; apply softmax to get probabilities:
```python
value_probs = scipy.special.softmax(result["value"][0])
win_prob = value_probs[0]
```

#### `ownership` (1, 1, 19, 19) - Float32

Predicted final territory ownership for each intersection:
- **Positive values**: Current player's territory
- **Negative values**: Opponent's territory
- **Near zero**: Neutral or dame points

Values typically range from -1.0 to +1.0.

**Usage**: Visualize territory predictions:
```python
ownership_map = result["ownership"][0, 0]
# Values > 0.5: Strongly current player's territory
# Values < -0.5: Strongly opponent's territory
```

#### `score_value` (1, 6) - Float32

Score distribution predictions (6 buckets). These represent the estimated final score after the game ends.

The exact bucket ranges depend on training, but typically cover score differences from large losses to large wins.

---

## Troubleshooting

### Build Issues

#### `conda: command not found`

**Problem**: Conda is not in your PATH or shell integration failed.

**Solution**:
```bash
# Restart your terminal
source ~/.zshrc

# Verify conda is accessible
conda --version

# If still not found, add to PATH manually
export PATH="$HOME/miniconda3/bin:$PATH"
```

#### `CMake not found`

**Problem**: CMake is not installed or not in PATH.

**Solution**:
```bash
# Install cmake via conda
conda install cmake

# Or install via Homebrew
brew install cmake
```

#### `Could not find numpy include path`

**Problem**: Numpy headers are not accessible during build.

**Solution**:
```bash
# Reinstall numpy in the conda environment
conda activate $HOME/miniconda3
conda install numpy

# Or let the build script handle it
make clean
make build
```

#### Build fails with "C++ compiler error"

**Problem**: Xcode Command Line Tools not installed or outdated.

**Solution**:
```bash
# Reinstall Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install

# Verify compiler
gcc --version
```

#### "libcoremlpython.so: Symbol not found"

**Problem**: C++ extensions were not built correctly.

**Solution**:
```bash
cd ~/katago_workspace/coremltools
make clean
make build

# If problem persists, check for mixed architectures (arm64/x86_64)
file coremltools/libcoremlpython.so
# Should match your CPU architecture
```

### Conversion Issues

#### `Only KataGo model versions (15, 16) are supported, got version X`

**Problem**: The model file is an older or newer version not supported by the converter.

**Solution**:
- Download a version 15 or 16 model from https://katagotraining.org/networks/
- Look for recent `kata1` models (2023-2024 training runs)
- Model filename usually indicates training date/version

#### `Could not decompress .bin.gz file`

**Problem**: The downloaded file is corrupted or incomplete.

**Solution**:
```bash
# Test if file is valid gzip
gunzip -t kata1-*.bin.gz

# If corrupt, re-download
rm kata1-*.bin.gz
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/[model-name].bin.gz

# Verify download size matches expected
ls -lh kata1-*.bin.gz
```

#### `Import error: cannot import name 'convert'`

**Problem**: The KataGo converter module is not found.

**Solution**:
```bash
# Verify converter files exist
ls coremltools/converters/katago/_converter.py

# If missing, re-clone repository
cd ~/katago_workspace
rm -rf coremltools
git clone https://github.com/ChinChangYang/coremltools.git
cd coremltools
git checkout katagocoremltools

# Rebuild
make build
```

#### Conversion hangs or takes extremely long

**Problem**: Model is very large or system is low on memory.

**Solution**:
- Close other applications to free memory
- Use a smaller model (fewer blocks/channels)
- Conversion time scales with model size:
  - Small (128ch, 10b): ~2-3 minutes
  - Medium (384ch, 20b): ~4-6 minutes
  - Large (512ch, 28b): ~6-10 minutes
  - Extra large (512ch, 40b): ~10-15 minutes

### Runtime Issues

#### `No module named 'coremltools.libcoremlpython'`

**Problem**: C++ extensions not built or not in Python path.

**Solution**:
```bash
cd ~/katago_workspace/coremltools

# Clean and rebuild
make clean
make build

# Ensure environment is activated
source scripts/env_activate.sh --python=3.11

# Verify extensions exist
ls coremltools/libcoremlpython.so
ls coremltools/libmilstoragepython.so
```

#### `MLModelError: Failed to load model`

**Problem**: .mlpackage is corrupted or incompatible.

**Solution**:
```bash
# Remove and reconvert
rm -rf KataGo.mlpackage
python convert_katago.py

# Verify package structure
ls -R KataGo.mlpackage/
```

#### Memory errors during inference

**Problem**: Insufficient RAM for model.

**Solution**:
- Close other applications
- Use a smaller model
- Check Activity Monitor for memory pressure
- Typical memory usage:
  - Model load: ~500MB-1GB
  - Inference: ~300MB-600MB additional
  - Total: ~1-2GB for typical models

#### Inference is very slow

**Problem**: Not using Neural Engine or optimizations disabled.

**Potential causes**:
```python
# Check compute units
mlmodel = ct.models.MLModel("KataGo.mlpackage")
print(mlmodel.compute_unit)

# Should use Neural Engine on Apple Silicon
# If using CPU_ONLY, performance will be slower

# For best performance on Apple Silicon:
mlmodel = ct.models.MLModel(
    "KataGo.mlpackage",
    compute_units=ct.ComputeUnit.ALL  # Uses Neural Engine + GPU + CPU
)
```

**Expected inference times** (Apple Silicon M1/M2):
- First inference (cold): ~500ms (includes compilation)
- Warm inference (Neural Engine): ~50-100ms
- CPU-only inference: ~200-500ms

---

## Future Work: KataGo Core ML Backend Integration

The converted Core ML model can be integrated into KataGo as a custom backend, allowing KataGo to use optimized Core ML inference on Apple Silicon devices.

**Benefits:**
- Native Apple Neural Engine acceleration
- Reduced power consumption vs GPU
- Simplified deployment (no CUDA/OpenCL setup)

**Requirements:**
- KataGo source code modifications

**Implementation status**: Future work (not yet implemented)

---

## Verification Checklist

Use this checklist to verify your setup is complete and working:

```bash
# 1. Check conda is installed and in PATH
conda --version
# ✓ Should show: conda 24.x.x or later

# 2. Check git
git --version
# ✓ Should show: git version 2.x.x or later

# 3. Check repository location
ls ~/katago_workspace/coremltools
# ✓ Should show coremltools directory contents

# 4. Check KataGo converter files
ls ~/katago_workspace/coremltools/coremltools/converters/katago/
# ✓ Should show: __init__.py, _converter.py, _katago_parser.py, etc.

# 5. Activate conda environment
cd ~/katago_workspace/coremltools
source scripts/env_activate.sh --python=3.11

# 6. Check Python version
python --version
# ✓ Should show: Python 3.11.x

# 7. Check coremltools import
python -c "import coremltools; print(coremltools.__version__)"
# ✓ Should output: 9.0

# 8. Check KataGo converter import
python -c "from coremltools.converters import katago; print('OK')"
# ✓ Should output: OK

# 9. Check model file exists
ls ~/katago_workspace/kata1-*.bin.gz
# ✓ Should show the downloaded .bin.gz file (~271MB)

# 10. Check converted model exists
ls ~/katago_workspace/KataGo.mlpackage/
# ✓ Should show directory structure (Data/, Manifest.json)

# 11. Test inference
cd ~/katago_workspace
python test_inference.py
# ✓ Should output: "Inference test PASSED!"
```

**All checks passing?** You're ready to use the KataGo to Core ML converter!

---

## Appendix

### Model File Formats

**KataGo Binary Format (.bin, .bin.gz)**:
- Custom binary format with header and layer data
- Gzip compression optional (recommended for distribution)
- Version markers (v15, v16, etc.)
- Contains model architecture and all weights

**Core ML Package (.mlpackage)**:
- Apple's standard ML model format
- Bundle directory containing:
  - `Manifest.json` - Package metadata
  - `Data/com.apple.CoreML/model.mlmodel` - Architecture
  - `Data/com.apple.CoreML/weights/` - Weight files
- Can be loaded directly in Swift/Objective-C with Core ML
- Optimized for Apple Silicon Neural Engine

### Supported KataGo Features

**✅ Supported:**
- Model versions 15 and 16
- Board size: 19x19
- All block types:
  - Ordinary residual blocks
  - Global pooling blocks
  - Nested bottleneck blocks
- Activation functions: Identity, ReLU, Mish
- Batch normalization (pre-merged scale/bias)
- Policy head (2 or 4 channels depending on version)
- Value head
- Ownership prediction
- Score value distribution

**❌ Not Currently Supported:**
- Model versions < 15 or > 16
- Variable board sizes (9x9, 13x13, etc.)
- SGF metadata encoder (skipped during conversion)
- Training-specific features

### Performance Considerations

**Batch Inference:**
Core ML models generated by this converter use batch size 1. For multiple positions:
- Run inference sequentially (simple but slower)
- Or modify model builder to support dynamic batching (advanced)

**Deployment Target:**
- `iOS15` (default): Maximum compatibility
- `iOS16+`: Access to newer Core ML features
- `macOS12+`: Required for Neural Engine on Mac

### Related Resources

**KataGo:**
- Official site: https://github.com/lightvector/KataGo
- Networks: https://katagotraining.org/networks/
- Model format: See `cpp/neuralnet/desc.h` in KataGo source

**Core ML:**
- coremltools docs: https://coremltools.readme.io/
- Core ML guide: https://developer.apple.com/documentation/coreml
- Neural Engine: https://machinelearning.apple.com/

**This Converter:**
- Repository: https://github.com/ChinChangYang/coremltools
- Branch: `katagocoremltools`
- Issues: Submit via GitHub Issues

---

## License

This converter implementation follows the coremltools BSD-3-Clause license.

KataGo models and networks are released under permissive licenses by the KataGo project.

---

**Questions or Issues?**

For problems with:
- **Converter**: Open GitHub issue at https://github.com/ChinChangYang/coremltools
- **KataGo models**: See https://github.com/lightvector/KataGo/issues
- **Core ML general**: See https://developer.apple.com/forums/tags/core-ml
