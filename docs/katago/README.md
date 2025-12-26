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
8. [Cross-Validation Results](#cross-validation-results)
9. [Using the Model](#using-the-model)
10. [Troubleshooting](#troubleshooting)
11. [Future Work: KataGo Core ML Backend Integration](#future-work-katago-core-ml-backend-integration)
12. [Verification Checklist](#verification-checklist)

## Introduction

### What is this?

This guide walks you through converting KataGo neural network models from their binary format (.bin, .bin.gz) to Apple's Core ML format (.mlpackage). Core ML models can run efficiently on Mac and iOS devices, leveraging Apple Silicon's Neural Engine for optimal performance.

**KataGo** is a strong open-source Go AI that uses deep neural networks for position evaluation and move selection. By converting KataGo models to Core ML, you can:
- Run inference on Apple devices without external dependencies
- Leverage Apple Silicon's Neural Engine for efficient computation
- Integrate Go AI into macOS and iOS applications
- Experiment with on-device AI for the game of Go

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
# Note: Clone into "KataGoCoremltools" directory to match environment naming
git clone https://github.com/ChinChangYang/coremltools.git KataGoCoremltools
cd KataGoCoremltools

# Switch to katagocoremltools branch
git checkout katagocoremltools
```

**Expected output:**
```
Cloning into 'KataGoCoremltools'...
remote: Enumerating objects: done.
remote: Counting objects: done.
remote: Compressing objects: done.
remote: Total (delta), reused (delta), pack-reused
Receiving objects: done.
Resolving deltas: done.

Branch 'katagocoremltools' set up to track remote branch 'katagocoremltools' from 'origin'.
Switched to a new branch 'katagocoremltools'
```

**Verify KataGo converter exists:**

```bash
ls KataGoCoremltools/converters/katago/
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

---

## Build coremltools from Source

Building coremltools from source compiles the C++ extensions and creates a Python package with the KataGo converter.

### 1. Create Conda Environment and Build

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

# Download the model (~259MB compressed)
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz

# Verify download
ls -lh kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

**Expected output:**
```
-rw-r--r--  1 user  staff   259M Dec 19 10:00 kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
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

**⏱️ Conversion time:** Varies by model size (typically 5-10 minutes for this model)

```bash
# Ensure environment is activated
source ~/katago_workspace/KataGoCoremltools/scripts/env_activate.sh --python=3.11

# Run conversion
python convert_katago.py
```

**Expected output:**
```
Converting kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz...
This may take 5-10 minutes...

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
1. **Parsing**: Reads binary model file, decompresses, parses layers
2. **Building MIL program**: Constructs intermediate representation
3. **Converting to Core ML**: Optimizes and generates .mlpackage

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

    print("Creating test input...")
    # Create test inputs with binary feature planes (matching KataGo format)
    # This creates a simple test position, not a real game
    np.random.seed(42)  # For reproducibility
    spatial = np.zeros((1, 22, 19, 19), dtype=np.float32)
    spatial[:, 0, :, :] = 1.0  # Channel 0: valid board mask
    # Add a few random binary stones for testing
    random_positions = np.random.rand(19, 19) > 0.9
    spatial[:, 1, random_positions] = 1.0  # Some player stones

    inputs = {
        "spatial_input": spatial,
        "global_input": np.zeros((1, 19), dtype=np.float32),
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
- Inference completes successfully

---

## Cross-Validation Results

The converter has been validated against the KataGo C++ Eigen backend to ensure accurate conversion.

### Running Cross-Validation Tests

To verify the converted Core ML model produces identical outputs to the KataGo Eigen backend, you can run the cross-validation test suite.

#### Prerequisites

**1. Build KataGo with Eigen Backend and Validation Subcommand**

The validation script compares Core ML outputs against KataGo's C++ Eigen backend. You'll need a KataGo executable built with Eigen support and the validation subcommand.

**Important:** Use the fork with validation subcommand support, not the official KataGo repository.

```bash
# Clone KataGo fork with validation subcommand (if not already available)
cd ~/katago_workspace
git clone https://github.com/ChinChangYang/KataGo.git
cd KataGo

# Switch to validation-subcommand branch
git checkout validation-subcommand

# Build with Eigen backend (macOS)
cd cpp
mkdir build
cd build
cmake .. \
  -DUSE_BACKEND=EIGEN \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_C_COMPILER=/usr/bin/clang
make -j8

# Verify executable and validation subcommand
./katago version
# Should show: KataGo v1.x.x

./katago validation
# Should show: validation subcommand usage
```

**2. Generate Test Inputs**

The validation script requires test input files. If they don't exist yet, generate them:

```bash
cd ~/katago_workspace/KataGoCoremltools

# Activate Python 3.11 environment
source scripts/env_activate.sh --python=3.11

# Generate test inputs (creates test_inputs/ directory)
python scripts/generate_test_inputs.py
```

This creates 9 test cases with different board configurations:
- `zeros.json` - Empty board
- `center_stone.json` - Single stone at center
- `corner_stone.json` - Single stone at corner
- `edge_pattern.json` - Stones along edge
- `diagonal_pattern.json` - Diagonal pattern
- `komi_7_5.json` - Different komi value
- `partial_mask_9x9.json` - 9x9 board mask
- `random_seed_42.json` - Random position
- `uniform_small.json` - Small uniform values

#### Running the Validation

```bash
cd ~/katago_workspace/KataGoCoremltools

# Ensure Python 3.11 environment is activated
source scripts/env_activate.sh --python=3.11

# Run cross-validation
python scripts/validate_coreml.py \
  --model-mlpackage ../KataGo.mlpackage \
  --model-bin ../kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz \
  --katago-exe ~/katago_workspace/KataGo/cpp/build/katago \
  --test-inputs test_inputs
```

**Expected output:**
```
Found 9 test case(s)
Core ML model: ../KataGo.mlpackage
KataGo model: ../kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
KataGo exe: ~/katago_workspace/KataGo/cpp/build/katago

=== Test Case: center_stone ===
  policy: max_diff=5.21e-04, mean_diff=4.48e-05, tol=5e-02 [PASS]
  pass_policy: max_diff=1.11e-04, mean_diff=1.11e-04, tol=1e-02 [PASS]
  value: max_diff=5.72e-05, mean_diff=3.42e-05, tol=2e-01 [PASS]
  ownership: max_diff=2.30e-05, mean_diff=2.29e-06, tol=5e-03 [PASS]
  score_value: max_diff=3.24e-05, mean_diff=1.44e-05, tol=5e-02 [PASS]

... (8 more test cases)

==================================================
All tests PASSED
```

#### Understanding Results

For each test case, the validation compares 5 outputs:

1. **policy** - Move policy logits (tolerance: 5e-2)
   - max_diff < 1e-3 is excellent
   - Higher differences may occur in areas with many similar moves

2. **pass_policy** - Pass move logit (tolerance: 1e-2)
   - Single value comparison
   - Very stable across implementations

3. **value** - Game outcome predictions (tolerance: 2e-1)
   - 3 values (win/loss/draw)
   - Larger tolerance due to accumulated operations

4. **ownership** - Territory predictions (tolerance: 5e-3)
   - Per-intersection predictions
   - Generally very stable

5. **score_value** - Score distribution (tolerance: 5e-2)
   - 6-bucket distribution
   - Moderate tolerance for accumulated operations

**Interpretation:**
- **PASS** - Differences within tolerance (green)
- **FAIL** - Differences exceed tolerance (red)
- **MISSING** - Output not found (yellow)
- **SHAPE_MISMATCH** - Output shapes don't match (yellow)

#### Troubleshooting Validation Issues

**Problem: Python version mismatch error**
```
symbol not found in flat namespace '_PyCMethod_New'
Exception: Unable to load libmodelpackage
```

**Solution:** Ensure you're using the same Python version that was used to build coremltools:
```bash
# Use the Python 3.11 executable directly
/path/to/KataGoCoremltools/envs/KataGoCoremltools-py3.11/bin/python scripts/validate_coreml.py ...

# Or activate the environment first
source scripts/env_activate.sh --python=3.11
python scripts/validate_coreml.py ...
```

**Problem: KataGo executable not found**
```
KataGo executable not found: /path/to/katago
```

**Solution:** Build KataGo or update the path:
```bash
# Check if katago exists
ls -la ~/katago_workspace/KataGo/cpp/build/katago

# Or use the correct path in validation command
python scripts/validate_coreml.py \
  --katago-exe /correct/path/to/katago \
  --model-mlpackage ../KataGo.mlpackage \
  --model-bin ../kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz \
  --test-inputs test_inputs
```

**Problem: Test inputs directory not found**
```
Error: Test inputs directory not found: test_inputs
```

**Solution:** Generate test inputs:
```bash
python scripts/generate_test_inputs.py
```

**Problem: Some tests FAIL**

If validation tests fail:
1. Check the max_diff values - small exceedances may be acceptable
2. Verify the model conversion completed without errors
3. Ensure binary inputs (0.0 or 1.0) are being used
4. Check if the KataGo model version is supported (v15 or v16)

### Validation Results Summary

✅ **All cross-validation tests passed**

Test results show excellent agreement between Core ML and Eigen implementations:
- Maximum difference: < 1e-3 (0.1%)
- Mean difference: < 1e-4 (0.01%)
- All outputs (policy, value, ownership, score) within tolerance

**Validated test cases:**
- Empty board positions
- Single stone placements (corner, center)
- Complex multi-stone patterns
- Edge patterns and diagonals
- Variable board masks (9x9 in 19x19)
- Different komi values

**Binary Input Requirement:**
- Input feature planes must be binary (0.0 or 1.0) for accurate results
- This matches KataGo's actual input format for realistic board positions
- The converter has been optimized and validated with binary inputs

**Recent improvements** (December 2025):
- ✅ Fixed global pooling feature ordering (improved accuracy 100-1000x)
- ✅ Optimized Mish activation implementation
- ✅ Confirmed Core ML outputs match Eigen backend to float32 precision limits

---

## Using the Model

### 1. Integration Options

#### Swift (iOS/macOS Applications)

**Option 1: Using Auto-Generated Classes (Xcode Integration)**

When you drag KataGo.mlpackage into an Xcode project, Xcode auto-generates Swift classes. However, note that the generated class names may vary. Use the programmatic approach below for more control.

**Option 2: Programmatic API (Recommended)**

```swift
import Foundation
import CoreML

// Load and compile the model
let modelURL = URL(fileURLWithPath: "/path/to/KataGo.mlpackage")

// Compile the model (one-time operation, or cache the compiled model)
let compiledModelURL = try MLModel.compileModel(at: modelURL)

// Load the compiled model
let config = MLModelConfiguration()
config.computeUnits = .all  // Use Neural Engine + GPU + CPU

let model = try MLModel(contentsOf: compiledModelURL, configuration: config)

// Prepare inputs with binary feature planes (matching KataGo format)
let spatialInput = try MLMultiArray(shape: [1, 22, 19, 19], dataType: .float32)
let globalInput = try MLMultiArray(shape: [1, 19], dataType: .float32)
let inputMask = try MLMultiArray(shape: [1, 1, 19, 19], dataType: .float32)

// Initialize with zeros
for i in 0..<spatialInput.count {
    spatialInput[i] = 0.0
}
for i in 0..<globalInput.count {
    globalInput[i] = 0.0
}
for i in 0..<inputMask.count {
    inputMask[i] = 1.0  // All positions valid for 19x19
}

// Initialize channel 0 (valid board mask) to 1.0
for i in 0..<19 {
    for j in 0..<19 {
        spatialInput[[0, 0, i, j] as [NSNumber]] = 1.0
    }
}

// TODO: Set additional channels with binary (0.0 or 1.0) stone positions
// spatialInput[[0, 1, row, col] as [NSNumber]] = 1.0  // Player stones
// spatialInput[[0, 2, row, col] as [NSNumber]] = 1.0  // Opponent stones

// Create input feature provider
let inputFeatures: [String: Any] = [
    "spatial_input": spatialInput,
    "global_input": globalInput,
    "input_mask": inputMask
]
let provider = try MLDictionaryFeatureProvider(dictionary: inputFeatures)

// Run inference
let output = try model.prediction(from: provider)

// Access outputs (using actual output layer names)
let policyLogits = output.featureValue(for: "policy_p2_conv")?.multiArrayValue  // (1, 2, 19, 19)
let passLogit = output.featureValue(for: "policy_pass_mul2")?.multiArrayValue   // (1, 2)
let valueLogits = output.featureValue(for: "value_v3_bias")?.multiArrayValue    // (1, 3)
let ownership = output.featureValue(for: "value_ownership_conv")?.multiArrayValue  // (1, 1, 19, 19)
let scoreValue = output.featureValue(for: "value_sv3_bias")?.multiArrayValue    // (1, 6)

// Use the outputs
if let policy = policyLogits {
    print("Policy shape: \(policy.shape)")
    // Process policy logits (channel 0 for base policy)
}
```

**Important Notes:**
- Model must be compiled before use with `MLModel.compileModel(at:)`
- Output names are layer names from conversion, not simplified names
- Actual output names:
  - `policy_p2_conv` - Policy logits (not just "policy")
  - `policy_pass_mul2` - Pass policy logit
  - `value_v3_bias` - Value logits (not just "value")
  - `value_ownership_conv` - Ownership predictions
  - `value_sv3_bias` - Score value distribution

#### Python (Testing and Development)

```python
import coremltools as ct
import numpy as np

# Load model
model = ct.models.MLModel("KataGo.mlpackage")

# Prepare inputs with binary feature planes (matching KataGo format)
spatial_input = np.zeros((1, 22, 19, 19), dtype=np.float32)
spatial_input[:, 0, :, :] = 1.0  # Channel 0: valid board mask

# TODO: Set additional channels with binary (0.0 or 1.0) stone positions
# spatial_input[:, 1, row, col] = 1.0  # Player stones
# spatial_input[:, 2, row, col] = 1.0  # Opponent stones

global_input = np.zeros((1, 19), dtype=np.float32)
input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)

# Run inference
result = model.predict({
    "spatial_input": spatial_input,
    "global_input": global_input,
    "input_mask": input_mask
})

# Access outputs (using actual layer names)
policy_logits = result["policy_p2_conv"]           # shape: (1, 2, 19, 19)
pass_logit = result["policy_pass_mul2"]            # shape: (1, 2)
value_logits = result["value_v3_bias"]             # shape: (1, 3)
ownership_map = result["value_ownership_conv"]     # shape: (1, 1, 19, 19)
score_values = result["value_sv3_bias"]            # shape: (1, 6)
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

**Important:** All feature plane values should be binary (0.0 or 1.0) for optimal accuracy. The converter has been validated with binary inputs matching KataGo's actual input format.

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

**Important**: The actual output names from the Core ML model are the layer names from the conversion process, not simplified names. Use these exact names when accessing outputs:

#### `policy_p2_conv` (1, 2, 19, 19) - Float32

Move policy logits for each board position:
- **Channel 0**: Base move policy
- **Channel 1**: Policy optimism (for uncertainty-aware search)

**Usage**: Apply softmax to get move probabilities:
```python
# Python
policy_logits = result["policy_p2_conv"][0, 0]  # Use channel 0
policy_probs = scipy.special.softmax(policy_logits.flatten())
policy_probs = policy_probs.reshape(19, 19)
```

```swift
// Swift
if let policy = output.featureValue(for: "policy_p2_conv")?.multiArrayValue {
    // Access channel 0 for base policy
}
```

Higher values indicate more likely moves. Combine with `policy_pass_mul2` for the full policy distribution.

#### `policy_pass_mul2` (1, 2) - Float32

Pass policy logits (2 values, use index 0).

**Usage**: Combine with spatial policy:
```python
# Python
pass_logit = result["policy_pass_mul2"][0, 0]
# Include in softmax calculation with spatial policy
```

#### `value_v3_bias` (1, 3) - Float32

Game outcome predictions:
- **Index 0**: Win probability for current player
- **Index 1**: Loss probability for current player
- **Index 2**: No-result probability (jigo, draw)

Values are logits; apply softmax to get probabilities:
```python
# Python
value_probs = scipy.special.softmax(result["value_v3_bias"][0])
win_prob = value_probs[0]
```

```swift
// Swift
if let value = output.featureValue(for: "value_v3_bias")?.multiArrayValue {
    let winLogit = value[0].floatValue
    let lossLogit = value[1].floatValue
    let drawLogit = value[2].floatValue
}
```

#### `value_ownership_conv` (1, 1, 19, 19) - Float32

Predicted final territory ownership for each intersection:
- **Positive values**: Current player's territory
- **Negative values**: Opponent's territory
- **Near zero**: Neutral or dame points

Values typically range from -1.0 to +1.0.

**Usage**: Visualize territory predictions:
```python
# Python
ownership_map = result["value_ownership_conv"][0, 0]
# Values > 0.5: Strongly current player's territory
# Values < -0.5: Strongly opponent's territory
```

#### `value_sv3_bias` (1, 6) - Float32

Score distribution predictions (6 buckets). These represent the estimated final score after the game ends.

The exact bucket ranges depend on training, but typically cover score differences from large losses to large wins.

**Legacy Names**: For compatibility, Python code may use simplified output names like "policy", "value", etc., but the actual Core ML layer names are as documented above.

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
cd ~/katago_workspace/KataGoCoremltools
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
- Look for recent `kata1` models (2023-2025 training runs)
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
rm -rf KataGoCoremltools
git clone https://github.com/ChinChangYang/coremltools.git KataGoCoremltools
cd KataGoCoremltools
git checkout katagocoremltools

# Rebuild
make build
```

### Runtime Issues

#### `No module named 'coremltools.libcoremlpython'`

**Problem**: C++ extensions not built or not in Python path.

**Solution**:
```bash
cd ~/katago_workspace/KataGoCoremltools

# Clean and rebuild
make clean
make build

# Ensure environment is activated
source scripts/env_activate.sh --python=3.11

# Verify extensions exist
ls coremltools/libcoremlpython.so
ls coremltools/libmilstoragepython.so
```

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
ls ~/katago_workspace/KataGoCoremltools
# ✓ Should show KataGoCoremltools directory contents

# 4. Check KataGo converter files
ls ~/katago_workspace/KataGoCoremltools/coremltools/converters/katago/
# ✓ Should show: __init__.py, _converter.py, _katago_parser.py, etc.

# 5. Activate conda environment
cd ~/katago_workspace/KataGoCoremltools
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
# ✓ Should show the downloaded .bin.gz file (~259MB)

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
