# KataGo to Core ML - Troubleshooting Guide

Comprehensive troubleshooting guide for common issues when converting KataGo models to Core ML format.

## Table of Contents

1. [Prerequisites Issues](#prerequisites-issues)
2. [Build Issues](#build-issues)
3. [Conversion Issues](#conversion-issues)
4. [Runtime Issues](#runtime-issues)
5. [Performance Issues](#performance-issues)
6. [Debugging Techniques](#debugging-techniques)

---

## Prerequisites Issues

### Conda Command Not Found

**Symptom:**
```bash
$ conda --version
zsh: command not found: conda
```

**Cause:**
Shell hasn't been restarted after Miniconda installation, or conda init wasn't run.

**Solutions:**

1. **Restart your terminal** (simplest solution):
   ```bash
   # Close and reopen Terminal app
   ```

2. **Source your shell configuration**:
   ```bash
   source ~/.zshrc
   conda --version
   ```

3. **Manually initialize conda**:
   ```bash
   ~/miniconda3/bin/conda init zsh
   source ~/.zshrc
   ```

4. **Add conda to PATH manually** (if above doesn't work):
   ```bash
   echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   ```

**Verification:**
```bash
conda --version
# Should output: conda 24.x.x
```

---

### Xcode Command Line Tools Not Installed

**Symptom:**
```bash
$ make build
xcrun: error: invalid active developer path (/Library/Developer/CommandLineTools)
```

**Cause:**
Xcode Command Line Tools not installed or corrupted.

**Solution:**
```bash
# Install Command Line Tools
xcode-select --install

# If already installed but corrupted, reset
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install

# Verify installation
xcode-select -p
# Should output: /Library/Developer/CommandLineTools

gcc --version
# Should show: Apple clang version...
```

---

### Git Not Available

**Symptom:**
```bash
$ git clone ...
zsh: command not found: git
```

**Solution:**
```bash
# Install Xcode Command Line Tools (includes git)
xcode-select --install

# Or install git separately via Homebrew
brew install git
```

---

## Build Issues

### CMake Not Found

**Symptom:**
```bash
CMake Error: Could not find CMAKE_ROOT !!!
```

**Solution:**
```bash
# Activate conda environment first
cd ~/katago_workspace/coremltools
source scripts/env_activate.sh --python=3.11

# Install cmake in conda environment
conda install cmake -y

# Verify
cmake --version
```

---

### Python Include Path Not Found

**Symptom:**
```bash
CMake Error: Could not find Python include path
```

**Cause:**
Python development headers not installed or not found by CMake.

**Solution:**
```bash
# Reinstall Python in conda environment
conda install python=3.11 -y

# Verify Python is from conda environment
which python
# Should show: .../coremltools/envs/KataGoCoremltools-py3.11/bin/python

python --version
# Should show: Python 3.11.x
```

---

### NumPy Import Error During Build

**Symptom:**
```bash
ModuleNotFoundError: No module named 'numpy'
```

**Solution:**
```bash
# Ensure environment is activated
source scripts/env_activate.sh --python=3.11

# Reinstall numpy
pip install numpy

# Verify
python -c "import numpy; print(numpy.__version__)"
```

---

### C++ Compilation Errors

**Symptom:**
```bash
fatal error: 'iostream' file not found
#include <iostream>
         ^~~~~~~~~~
```

**Cause:**
Missing or corrupted C++ standard library headers.

**Solution:**
```bash
# Reinstall Xcode Command Line Tools
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install

# Reset Xcode path
sudo xcode-select --reset
```

---

### Architecture Mismatch (Intel Mac)

**Symptom:**
```bash
ld: warning: ignoring file libcoremlpython.so, building for macOS-arm64 but attempting to link with file built for macOS-x86_64
```

**Cause:**
Build scripts are configured for arm64 (Apple Silicon), but you're on Intel Mac.

**Solution:**
```bash
# Edit scripts/build.sh
# Change line ~115 from:
-DCMAKE_OSX_ARCHITECTURES=arm64

# To:
-DCMAKE_OSX_ARCHITECTURES=x86_64

# Clean and rebuild
make clean
make build
```

---

### Build Hangs or Takes Too Long

**Symptom:**
Build process appears stuck or takes more than 90 minutes.

**Causes:**
- Insufficient memory
- Too many parallel jobs
- Swap thrashing

**Solutions:**

1. **Check available memory**:
   ```bash
   vm_stat | head -n 10
   ```

2. **Reduce parallel jobs**:
   ```bash
   # Edit scripts/build.sh
   # Find the make command and add -j flag
   make -j2  # Use only 2 parallel jobs instead of all cores
   ```

3. **Free up memory**:
   ```bash
   # Close other applications
   # Restart terminal
   # Try build again
   make clean
   make build
   ```

4. **Monitor build progress**:
   ```bash
   # Build with verbose output
   make build VERBOSE=1
   ```

---

### libcoremlpython Not Found

**Symptom:**
```bash
ImportError: dlopen(...libcoremlpython.so): image not found
```

**Cause:**
C++ extensions weren't built properly or Python can't find them.

**Solutions:**

1. **Rebuild from clean state**:
   ```bash
   cd ~/katago_workspace/coremltools
   make clean
   make build
   ```

2. **Verify build directory**:
   ```bash
   ls build/coremltools
   # Should contain libcoremlpython.so or similar
   ```

3. **Check environment activation**:
   ```bash
   source scripts/env_activate.sh --python=3.11
   python -c "import coremltools; print(coremltools.__version__)"
   ```

---

### Environment Already Exists Error

**Symptom:**
```bash
CondaValueError: prefix already exists: .../envs/KataGoCoremltools-py3.11
```

**Solutions:**

1. **Remove existing environment**:
   ```bash
   conda env remove -p envs/KataGoCoremltools-py3.11
   make build
   ```

2. **Force recreate**:
   ```bash
   rm -rf envs/KataGoCoremltools-py3.11
   make build
   ```

---

## Conversion Issues

### Model Version Not Supported

**Symptom:**
```bash
ValueError: Only KataGo model versions (15, 16) are supported, but got version 14
```

**Cause:**
Model file is too old (version 14 or earlier).

**Solution:**
Download a newer model from https://katagotraining.org/networks/

```bash
# Example: Download a v15/v16 model
cd ~/katago_workspace
curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz

# Verify it's a supported version by checking the conversion
python convert_katago.py
```

**How to check model version manually:**
```bash
# Decompress and check binary header
gunzip -c kata1-*.bin.gz | head -c 100
# Look for version number in output
```

---

### Decompression Failed

**Symptom:**
```bash
gzip.BadGzipFile: Not a gzipped file
```

**Causes:**
- Incomplete download
- Corrupted file
- File is already decompressed

**Solutions:**

1. **Verify file integrity**:
   ```bash
   gunzip -t kata1-*.bin.gz
   # If fails, re-download
   ```

2. **Check file size**:
   ```bash
   ls -lh kata1-*.bin.gz
   # Should be ~270-300MB for most models
   ```

3. **Re-download if corrupted**:
   ```bash
   rm kata1-*.bin.gz
   curl -O https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz
   ```

4. **If file is already decompressed** (.bin instead of .bin.gz):
   ```python
   # Edit convert_katago.py to use .bin instead of .bin.gz
   model_path = "kata1-b28c512nbt-adam-s11165M-d5387M.bin"
   ```

---

### KataGo Converter Import Error

**Symptom:**
```bash
ImportError: cannot import name 'convert' from 'coremltools.converters.katago'
```

**Cause:**
KataGo converter not present in the codebase or not properly built.

**Solutions:**

1. **Verify branch**:
   ```bash
   cd ~/katago_workspace/coremltools
   git branch
   # Should show: * katagocoremltools
   ```

2. **Check converter files exist**:
   ```bash
   ls coremltools/converters/katago/
   # Should show: __init__.py, _converter.py, _katago_parser.py, etc.
   ```

3. **Rebuild coremltools**:
   ```bash
   make clean
   make build
   source scripts/env_activate.sh --python=3.11
   ```

4. **Verify import**:
   ```bash
   python -c "from coremltools.converters import katago; print(dir(katago))"
   # Should show: ['convert', ...]
   ```

---

### Conversion Hangs

**Symptom:**
Conversion process appears stuck with no progress for >15 minutes.

**Causes:**
- Memory exhaustion
- Swap thrashing
- Very large model

**Solutions:**

1. **Monitor memory usage**:
   ```bash
   # In another terminal
   top -o MEM
   # Watch Python process memory usage
   ```

2. **Close other applications**:
   ```bash
   # Free up RAM
   # Close browsers, IDEs, etc.
   ```

3. **Try a smaller model**:
   ```bash
   # Download a smaller model for testing
   # Look for models with fewer blocks (e.g., b10c256)
   ```

4. **Add progress monitoring**:
   ```python
   # Edit convert_katago.py
   import logging
   logging.basicConfig(level=logging.INFO)

   mlmodel = ct.converters.katago.convert(
       model_path,
       minimum_deployment_target=ct.target.iOS15,
       verbose=True  # If supported
   )
   ```

---

### Conversion Fails with Shape Mismatch

**Symptom:**
```bash
ValueError: Shape mismatch in layer 'xyz': expected (1, 512, 19, 19), got (1, 256, 19, 19)
```

**Cause:**
Bug in converter or unsupported model architecture variant.

**Solutions:**

1. **Check model architecture**:
   ```bash
   # Parse model to inspect architecture
   python test_parser_standalone.py
   # Review output for unusual layer configurations
   ```

2. **Try different model**:
   ```bash
   # Download a standard architecture model
   # Avoid experimental or custom architectures
   ```

3. **Report issue**:
   ```bash
   # If you believe this is a bug, report it with:
   # - Model file name and URL
   # - Full error traceback
   # - Output of test_parser_standalone.py
   ```

---

## Runtime Issues

### Model Loading Fails

**Symptom:**
```bash
RuntimeError: Failed to load model: KataGo.mlpackage
```

**Causes:**
- Corrupted .mlpackage
- Incomplete conversion
- Incompatible macOS version

**Solutions:**

1. **Verify .mlpackage structure**:
   ```bash
   ls -la KataGo.mlpackage/
   # Should contain: Data/, Metadata/, model.mlmodel or similar
   ```

2. **Check macOS version**:
   ```bash
   sw_vers
   # ProductVersion should be >= 12.0 for iOS15 deployment target
   ```

3. **Reconvert with different deployment target**:
   ```python
   # Edit convert_katago.py
   mlmodel = ct.converters.katago.convert(
       model_path,
       minimum_deployment_target=ct.target.iOS14  # Try older target
   )
   ```

4. **Verify with Core ML tools**:
   ```bash
   python -c "import coremltools as ct; model = ct.models.MLModel('KataGo.mlpackage'); print(model)"
   ```

---

### Prediction Fails with Input Shape Error

**Symptom:**
```bash
RuntimeError: Unexpected input shape for 'spatial_input': expected (1, 22, 19, 19), got (1, 19, 19, 22)
```

**Cause:**
Input arrays have incorrect shape or order.

**Solution:**
Verify input shapes match exactly:

```python
import numpy as np

# CORRECT shapes:
spatial_input = np.random.randn(1, 22, 19, 19).astype(np.float32)
global_input = np.random.randn(1, 19).astype(np.float32)
input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)

# Verify shapes before prediction
print(f"spatial_input: {spatial_input.shape}")  # (1, 22, 19, 19)
print(f"global_input: {global_input.shape}")    # (1, 19)
print(f"input_mask: {input_mask.shape}")        # (1, 1, 19, 19)

result = mlmodel.predict({
    "spatial_input": spatial_input,
    "global_input": global_input,
    "input_mask": input_mask
})
```

---

### Prediction Returns NaN or Inf

**Symptom:**
```python
# Output contains NaN or Inf values
result['policy'][0, 0, 0, 0]  # nan
```

**Causes:**
- Invalid input values
- Numerical instability
- Model conversion issue

**Solutions:**

1. **Check input validity**:
   ```python
   # Ensure inputs have no NaN/Inf
   assert not np.isnan(spatial_input).any()
   assert not np.isinf(spatial_input).any()
   assert not np.isnan(global_input).any()
   assert not np.isinf(global_input).any()
   ```

2. **Use reasonable input ranges**:
   ```python
   # Avoid extreme values
   spatial_input = np.random.randn(1, 22, 19, 19).astype(np.float32) * 0.1
   global_input = np.random.randn(1, 19).astype(np.float32) * 0.1
   ```

3. **Test with zeros**:
   ```python
   # Simplest valid input
   spatial_input = np.zeros((1, 22, 19, 19), dtype=np.float32)
   global_input = np.zeros((1, 19), dtype=np.float32)
   input_mask = np.ones((1, 1, 19, 19), dtype=np.float32)
   ```

4. **Verify model conversion**:
   ```bash
   # Reconvert and test again
   python convert_katago.py
   python test_inference.py
   ```

---

### Slow Inference Performance

**Symptom:**
Inference takes >1 second per prediction.

**Causes:**
- Running on Intel Mac instead of Apple Silicon
- Using CPU instead of Neural Engine
- Debug build instead of optimized build

**Solutions:**

1. **Verify running on Apple Silicon**:
   ```bash
   uname -m
   # Should output: arm64

   sysctl -n machdep.cpu.brand_string
   # Should show: Apple M1/M2/M3...
   ```

2. **Check Core ML compute units**:
   ```python
   import coremltools as ct

   # Load model with specific compute units
   model = ct.models.MLModel(
       "KataGo.mlpackage",
       compute_units=ct.ComputeUnit.ALL  # Use all available (CPU, GPU, Neural Engine)
   )
   ```

3. **Profile inference**:
   ```python
   import time

   # Warm up
   for _ in range(3):
       model.predict(inputs)

   # Measure
   start = time.time()
   for _ in range(10):
       result = model.predict(inputs)
   avg_time = (time.time() - start) / 10
   print(f"Average inference time: {avg_time*1000:.2f} ms")
   ```

4. **Expected performance**:
   - Apple Silicon (M1/M2): 5-20ms
   - Intel Mac: 50-200ms

---

## Performance Issues

### Memory Usage Too High

**Symptom:**
Python process uses >4GB of RAM during inference.

**Solutions:**

1. **Release model after use**:
   ```python
   import coremltools as ct

   # Load model
   model = ct.models.MLModel("KataGo.mlpackage")

   # Use model
   result = model.predict(inputs)

   # Release model
   del model
   import gc
   gc.collect()
   ```

2. **Avoid loading multiple models**:
   ```python
   # BAD: Loading model in loop
   for i in range(100):
       model = ct.models.MLModel("KataGo.mlpackage")
       result = model.predict(inputs)

   # GOOD: Load once, reuse
   model = ct.models.MLModel("KataGo.mlpackage")
   for i in range(100):
       result = model.predict(inputs)
   ```

---

### Batch Inference Not Faster

**Symptom:**
Processing 10 inputs separately is as fast as batch of 10.

**Cause:**
Model was converted with fixed batch size 1.

**Note:**
Current converter creates models with fixed batch size 1. Batch inference requires reconversion with variable batch size support (future enhancement).

**Workaround:**
Process inputs in parallel using multiprocessing (on multi-core systems).

---

## Debugging Techniques

### Enable Verbose Logging

```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Run conversion
import coremltools as ct
mlmodel = ct.converters.katago.convert("model.bin.gz")
```

---

### Inspect Parsed Model Structure

```bash
# Use standalone parser test
python test_parser_standalone.py

# Review output to understand model architecture
```

---

### Validate Model Outputs

```python
import numpy as np
import coremltools as ct

model = ct.models.MLModel("KataGo.mlpackage")

# Create known inputs
inputs = {
    "spatial_input": np.zeros((1, 22, 19, 19), dtype=np.float32),
    "global_input": np.zeros((1, 19), dtype=np.float32),
    "input_mask": np.ones((1, 1, 19, 19), dtype=np.float32)
}

# Run prediction
result = model.predict(inputs)

# Validate outputs
for key, value in result.items():
    print(f"\n{key}:")
    print(f"  Shape: {value.shape}")
    print(f"  Dtype: {value.dtype}")
    print(f"  Min: {value.min()}")
    print(f"  Max: {value.max()}")
    print(f"  Has NaN: {np.isnan(value).any()}")
    print(f"  Has Inf: {np.isinf(value).any()}")
```

---

### Compare with Original Model

If you have the original KataGo engine:

```bash
# Run same position through both models
# Compare policy/value outputs
# Verify conversion correctness
```

---

### Check Core ML System Support

```python
import coremltools as ct

# Check available compute units
print(f"Core ML version: {ct.__version__}")
print(f"Available compute units: {ct.models.ComputeUnit}")

# Load model and check configuration
model = ct.models.MLModel("KataGo.mlpackage")
spec = model.get_spec()
print(f"Model type: {spec.WhichOneof('Type')}")
```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the main README**: [docs/katago/README.md](README.md)
2. **Review the quick start**: [docs/katago/QUICK_START.md](QUICK_START.md)
3. **Search existing issues**: https://github.com/ChinChangYang/coremltools/issues
4. **Create a new issue** with:
   - macOS version (`sw_vers`)
   - CPU architecture (`uname -m`)
   - Python version (`python --version`)
   - coremltools version (`python -c "import coremltools; print(coremltools.__version__)"`)
   - Model file name and source URL
   - Complete error message and traceback
   - Steps to reproduce

---

## Quick Reference: Common Error Messages

| Error Message | Section |
|---------------|---------|
| `conda: command not found` | [Conda Command Not Found](#conda-command-not-found) |
| `xcrun: error: invalid active developer path` | [Xcode Command Line Tools Not Installed](#xcode-command-line-tools-not-installed) |
| `CMake Error: Could not find CMAKE_ROOT` | [CMake Not Found](#cmake-not-found) |
| `ModuleNotFoundError: No module named 'numpy'` | [NumPy Import Error During Build](#numpy-import-error-during-build) |
| `libcoremlpython.so: image not found` | [libcoremlpython Not Found](#libcoremlpython-not-found) |
| `Only KataGo model versions (15, 16) are supported` | [Model Version Not Supported](#model-version-not-supported) |
| `gzip.BadGzipFile: Not a gzipped file` | [Decompression Failed](#decompression-failed) |
| `ImportError: cannot import name 'convert'` | [KataGo Converter Import Error](#katago-converter-import-error) |
| `RuntimeError: Failed to load model` | [Model Loading Fails](#model-loading-fails) |
| `RuntimeError: Unexpected input shape` | [Prediction Fails with Input Shape Error](#prediction-fails-with-input-shape-error) |

---

**Last updated**: 2025-12-19
