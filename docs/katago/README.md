# KataGo to Core ML Converter

This guide walks you through converting KataGo neural network models to Apple's Core ML format on macOS. By following these steps, you'll be able to run KataGo models efficiently on Mac and iOS devices using Apple Silicon's Neural Engine.

## What You'll Achieve

After completing this guide, you will have:
- A working coremltools installation with KataGo converter support
- A converted KataGo model in `.mlpackage` format ready for use on Apple devices
- Verified that the model produces correct inference results
- Understanding of how to validate the conversion accuracy

**Requirements:**
- macOS with Apple Silicon (M1/M2/M3/M4)
- Internet connection for downloading tools and models
- Approximately 1-2 GB of disk space

---

## Prerequisites Installation

### Install Xcode Command Line Tools

**What this does:** Xcode Command Line Tools provide essential development utilities including the C/C++ compilers (clang), build tools (make, cmake), and git version control. These are required to build the coremltools package from source.

```bash
xcode-select --install
```

**Expected result:** A dialog window will appear asking you to install the Command Line Tools. Click "Install" and accept the license agreement. The installation will download approximately 500MB and take 5-10 minutes to complete.

**Verify installation:**
```bash
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
# or: /Applications/Xcode.app/Contents/Developer

gcc --version
# Should output: Apple clang version 14.0.0 or later
```

If you see "command line tools are already installed" when running `xcode-select --install`, you're all set and can proceed to the next step.

### Install Miniconda

**What this does:** Miniconda is a lightweight Python distribution manager that creates isolated Python environments. We'll use it to set up a dedicated Python 3.11 environment for building coremltools without affecting your system Python or other projects.

**For Apple Silicon (M1/M2/M3/M4):**

```bash
# Download the installer (approximately 50MB)
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Install to ~/miniconda3 (default location)
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

# Initialize shell integration for zsh (macOS default shell)
~/miniconda3/bin/conda init zsh

# Restart terminal or reload configuration
source ~/.zshrc
```

**Expected result:** The installer will create a `~/miniconda3` directory containing the conda package manager and a base Python environment. After running `conda init zsh` and reloading your shell, you should see `(base)` appear in your command prompt, indicating conda is active.

**Verify installation:**
```bash
conda --version
# Should output: conda 24.x.x or later
```

If conda is not found after reloading your shell, you may need to manually add it to your PATH:
```bash
export PATH="$HOME/miniconda3/bin:$PATH"
```

---

## Setup and Build

### Clone the Repository

**What this does:** This downloads the coremltools fork that contains the KataGo converter. The converter is located in the `katagocoremltools` branch and includes all necessary code to parse KataGo binary models and convert them to Core ML format.

```bash
# Create a workspace directory for all KataGo-related files
mkdir -p ~/katago_workspace && cd ~/katago_workspace

# Clone the coremltools repository
git clone https://github.com/ChinChangYang/coremltools.git

# Enter the repository directory
cd coremltools

# Switch to the branch containing the KataGo converter
git checkout katagocoremltools
```

**Expected result:** Git will download the repository (approximately 100-200MB) and switch to the `katagocoremltools` branch. You should see messages like:
```
Cloning into 'coremltools'...
remote: Enumerating objects: done.
remote: Counting objects: 100% (xxx/xxx), done.
remote: Compressing objects: 100% (xxx/xxx), done.
remote: Total xxx (delta xxx), reused xxx (delta xxx), pack-reused xxx
Receiving objects: 100% (xxx/xxx), done.
Resolving deltas: 100% (xxx/xxx), done.
Branch 'katagocoremltools' set up to track remote branch 'katagocoremltools' from 'origin'.
Switched to a new branch 'katagocoremltools'
```

**Verify the KataGo converter is present:**
```bash
ls coremltools/converters/katago/
# Should show: __init__.py  _converter.py  _katago_model_builder.py  _katago_ops.py  _katago_parser.py  _katago_types.py
```

### Build the Package

**What this does:** The `make wheel` command builds coremltools from source, which involves:
1. Creating a Python 3.11 conda environment at `envs/coremltools-py3.11/`
2. Installing all Python dependencies (numpy, protobuf, etc.)
3. Compiling C++ extensions (`libcoremlpython`, `libmilstoragepython`) that provide Core ML functionality
4. Building a Python wheel (`.whl`) package in the `build/dist/` directory

```bash
make wheel
```

**Expected result:** The build process will take 30-60 seconds. You'll see extensive output as CMake configures the build, compilers process C++ files, and Python dependencies are installed. Key progress indicators:

```
Creating conda environment in envs/coremltools-py3.11
Solving environment: done
Installing pip dependencies...
Running CMake...
-- The CXX compiler identification is AppleClang 14.0.0
-- Configuring done
-- Generating done
-- Build files written to: .../build
[ 10%] Building CXX object deps/protobuf/...
[ 20%] Building CXX object mlmodel/...
...
[100%] Built target coremlpython
Successfully built coremltools-9.0-cp311-cp311-macosx_14_0_arm64.whl
```

### Activate the Environment

**What this does:** This script activates the Python 3.11 conda environment that was created during the build. It sets up environment variables so that Python commands use the correct interpreter and can find the coremltools modules.

```bash
source scripts/env_activate.sh --python=3.11
```

**Expected result:** Your shell prompt may change to show `(coremltools-py3.11)` or similar, indicating the conda environment is active. The environment variables `PATH`, `PYTHONPATH`, and related variables are now configured.

**Verify activation:**
```bash
which python
# Should output: .../coremltools/envs/coremltools-py3.11/bin/python

python --version
# Should output: Python 3.11.x
```

### Install the Wheel Package

**What this does:** This installs the coremltools wheel package that was built in the previous step. The wheel contains all the Python code and compiled C++ extensions needed to use coremltools, including the KataGo converter.

```bash
pip install build/dist/coremltools*cp311*arm64.whl
```

**Expected result:** Pip will install the wheel package, which typically takes 5-10 seconds. You'll see output like:
```
Processing ./build/dist/coremltools-9.0-cp311-cp311-macosx_14_0_arm64.whl
Installing collected packages: coremltools
Successfully installed coremltools-9.0
```

**Verify installation:**
```bash
# Navigate back to workspace to avoid import conflicts
cd ~/katago_workspace

# Check coremltools version
python -c "import coremltools; print(coremltools.__version__)"
# Should output: 9.0

# Verify KataGo converter is available
python -c "from coremltools.converters import katago; print('OK')"
# Should output: OK
```

If you see "OK", the KataGo converter is successfully installed and ready to use!

---

## Download a KataGo Model

**What this does:** This downloads a production-quality KataGo neural network model in binary format (`.bin.gz`). This specific model uses 28 nested bottleneck blocks with 512 channels and was trained on over 11 billion positions. It's a strong model at approximately 9-10 dan professional level.

**Model details:**
- **Version**: 15 (fully supported by the converter)
- **Architecture**: 28 nested bottleneck blocks, 512 channels
- **Board size**: 19x19 (only size currently supported)
- **File size**: ~259MB compressed

```bash
cd ~/katago_workspace
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

**Expected result:** The download will take 1-3 minutes depending on your internet connection. You'll see progress output:
```
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  259M  100  259M    0     0  15.2M      0  0:00:17  0:00:17 --:--:-- 18.3M
```

**Verify download:**
```bash
ls -lh kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
# Should show: -rw-r--r--  1 user  staff   259M Dec 26 10:00 kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
```

The file should be approximately 259MB. If the download was interrupted or the file size is significantly different, delete it and download again.

**Alternative models:** For other KataGo models, visit https://katagotraining.org/networks/ and look for `kata1-b*` files. The converter supports model versions 15 and 16 only.

---

## Convert Model to Core ML

### Create Conversion Script

**What this does:** This creates a Python script that will convert the KataGo binary model to Core ML's `.mlpackage` format. The converter reads the binary file, parses the neural network architecture and weights, and generates a Core ML model optimized for Apple Silicon's Neural Engine.

```bash
cd ~/katago_workspace

cat > convert_katago.py << 'EOF'
#!/usr/bin/env python3
import sys, coremltools as ct

mlmodel = ct.converters.katago.convert(
    "kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz",
    minimum_deployment_target=ct.target.iOS15
)
mlmodel.save("KataGo.mlpackage")
print("Conversion complete!")
EOF
```

**Expected result:** A file named `convert_katago.py` is created in your workspace directory. You can verify it was created:
```bash
ls -l convert_katago.py
# Should show: -rw-r--r--  1 user  staff  xxx Dec 26 10:00 convert_katago.py
```

The script uses `minimum_deployment_target=ct.target.iOS15` to ensure maximum compatibility with devices running iOS 15 or later, and macOS 12 or later.

### Run Conversion

**What this does:** Executes the conversion script. The converter will:
1. Decompress and parse the `.bin.gz` file
2. Extract model architecture (layers, blocks, activations) and weights
3. Build a Model Intermediate Language (MIL) program
4. Convert the MIL program to Core ML format
5. Save the result as `KataGo.mlpackage`

```bash
python convert_katago.py
```

**Expected result:** The conversion takes 10-20 seconds. You'll see output indicating progress:
```
Conversion complete!
```

**Verify output:**
```bash
# Check the package was created
ls KataGo.mlpackage/
# Should show: Data/  Manifest.json

# Check total size
du -sh KataGo.mlpackage/
# Should output: 250M-300M  KataGo.mlpackage/
```

The `.mlpackage` is a directory bundle containing:
- **Manifest.json** - Package metadata
- **Data/com.apple.CoreML/model.mlmodel** - Model architecture and metadata
- **Data/com.apple.CoreML/weights/weight.bin** - Neural network weights

If you see these files and the size is reasonable (250-300MB), the conversion was successful!

---

## Test the Converted Model

### Create Test Script

**What this does:** This creates a simple test script that loads the converted Core ML model and runs inference with random inputs. This verifies that:
1. The model can be loaded successfully
2. Inference runs without errors
3. Output shapes are correct
4. The model produces reasonable numerical outputs

```bash
cd ~/katago_workspace

cat > test_inference.py << 'EOF'
#!/usr/bin/env python3
import numpy as np, coremltools as ct

model = ct.models.MLModel("KataGo.mlpackage")
result = model.predict({
    "spatial_input": np.random.randn(1, 22, 19, 19).astype(np.float32),
    "global_input": np.random.randn(1, 19).astype(np.float32),
    "input_mask": np.ones((1, 1, 19, 19), dtype=np.float32)
})
print("Inference test PASSED!")
for key, val in result.items():
    if hasattr(val, 'shape'):
        print(f"  {key}: {val.shape}")
EOF
```

**Expected result:** A file named `test_inference.py` is created. This script uses random inputs (not a real board position) just to verify the model works.

**Input explanation:**
- **spatial_input** (1, 22, 19, 19): 22 feature planes representing board state
- **global_input** (1, 19): 19 global features (komi, rules, etc.)
- **input_mask** (1, 1, 19, 19): Binary mask for valid board positions

### Run Test

**What this does:** Executes the test script to verify the converted model works correctly.

```bash
python test_inference.py
```

**Expected result:** The test should complete in 1-2 seconds and produce output showing successful inference:
```
Inference test PASSED!
  policy_p2_conv: (1, 2, 19, 19)
  policy_pass_mul2: (1, 2)
  value_v3_bias: (1, 3)
  value_ownership_conv: (1, 1, 19, 19)
  value_sv3_bias: (1, 6)
```

**Output explanation:**
- **policy_p2_conv**: Move policy logits for each board position (2 channels)
- **policy_pass_mul2**: Pass move policy logit
- **value_v3_bias**: Game outcome predictions (win/loss/draw)
- **value_ownership_conv**: Territory ownership predictions
- **value_sv3_bias**: Score distribution predictions

If you see all five outputs with the correct shapes, the model is working correctly!

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
cd ~/katago_workspace/coremltools

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
cd ~/katago_workspace/coremltools

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

## Verification Checklist

Use these commands to verify your complete setup:

```bash
# 1. Check conda is installed and in PATH
conda --version
# ✓ Should show: conda 24.x.x or later

# 2. Check Python version in active environment
python --version
# ✓ Should show: Python 3.11.x

# 3. Check coremltools is installed
python -c "import coremltools; print(coremltools.__version__)"
# ✓ Should output: 9.0

# 4. Check KataGo converter is available
python -c "from coremltools.converters import katago; print('OK')"
# ✓ Should output: OK

# 5. Verify model file was downloaded
ls -lh kata1-*.bin.gz
# ✓ Should show: ~259MB file

# 6. Verify converted model exists
ls KataGo.mlpackage/
# ✓ Should show: Data/  Manifest.json

# 7. Run inference test
python test_inference.py
# ✓ Should output: "Inference test PASSED!" with 5 output shapes
```

If all checks pass, your KataGo to Core ML converter is fully set up and working correctly!

---

## File Structure Summary

After completing this guide, your workspace should look like this:

```
~/katago_workspace/
├── coremltools/                                     # Repository
│   ├── coremltools/converters/katago/               # Converter source code
│   ├── envs/coremltools-py3.11/                     # Conda environment
│   ├── build/dist/coremltools-9.0-*.whl             # Built wheel package
│   └── scripts/                                     # Build and activation scripts
├── kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz      # Input model (~259MB)
├── KataGo.mlpackage/                                # Output model (~250-300MB)
│   ├── Data/com.apple.CoreML/model.mlmodel          # Model architecture
│   ├── Data/com.apple.CoreML/weights/weight.bin     # Model weights
│   └── Manifest.json                                # Package metadata
├── convert_katago.py                                # Conversion script
└── test_inference.py                                # Inference test script
```

**File sizes:**
- Input model: ~259MB (compressed .bin.gz)
- Output model: ~250-300MB (.mlpackage directory)
- Total workspace: ~1-2GB including build artifacts

---

## Next Steps

Now that you have a working Core ML model, you can:

1. **Integrate into Swift applications** - Load the `.mlpackage` in Xcode for iOS/macOS apps
2. **Run performance benchmarks** - Compare inference speed on CPU vs Neural Engine
3. **Try other models** - Convert different KataGo models from https://katagotraining.org/networks/
4. **Validate accuracy** - Run the cross-validation tests to verify numerical correctness

For questions or issues with the converter, please open an issue at:
https://github.com/ChinChangYang/coremltools/issues

For general KataGo questions, see:
https://github.com/lightvector/KataGo

---

## License

This converter implementation follows the coremltools BSD-3-Clause license.

KataGo models and networks are released under permissive licenses by the KataGo project.
