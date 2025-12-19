# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

CoreMLTools is a Python package for converting machine learning models from third-party frameworks (TensorFlow, PyTorch, scikit-learn, XGBoost, etc.) to Apple's Core ML format. The codebase includes converters, an intermediate representation (MIL), optimization passes, and utilities for creating and manipulating .mlmodel files.

## Build Commands

### Environment Setup
```bash
# Create development environment (uses Python 3.7 by default)
make env

# Force recreate environment
make env_force

# Use different Python version
make env python=3.8  # or 3.9, 3.10, 3.11, 3.12, 3.13
```

### Building
```bash
# Build in debug mode (includes symbols)
make build

# Build with specific Python version
make build python=3.8

# Build wheels in release mode
make wheel

# Rebuild MLModel protobuf sources
make proto
```

### Testing
```bash
# Run all tests
make test

# Run only fast tests (exclude tests marked with @pytest.mark.slow)
make test_fast

# Run only slow tests
make test_slow

# Run specific test package
make test TEST_PACKAGES="coremltools.test.optimize"

# Run tests directly with pytest
pytest coremltools/test/optimize/torch/quantization/
```

Test markers (see pytest.ini):
- `@pytest.mark.slow` - Tests taking >1 second

### Code Quality
```bash
# Run linter (pylint)
make lint

# Run style checker (yapf)
make style

# Check style without making changes
make checkstyle
```

### Documentation
```bash
# Build documentation
make docs
```

### Cleanup
```bash
# Clean build directory
make clean

# Delete all environments
make clean_envs
```

## High-Level Architecture

### Conversion Pipeline

The model conversion follows this flow:

```
Source Model (TF/PyTorch/sklearn/etc)
    ↓
[FRONTEND] - Loads source model, converts to MIL Program
    ↓
[MIL Program] - Model Intermediate Language (unified representation)
    ↓
[OPTIMIZATION PASSES] - Graph optimizations via PassPipeline
    ↓
[BACKEND] - Converts MIL to target format (neuralnetwork or mlprogram)
    ↓
Target Model (.mlmodel or .mlpackage)
```

### Main Components

**Entry Point**: `coremltools/converters/_converters_entry.py`
- `convert()` function is the main user-facing API
- Routes to appropriate frontend based on source framework
- Orchestrates the entire conversion pipeline via `mil_convert()`

**Frontends** (`coremltools/converters/mil/frontend/`):
- `torch/` - PyTorch and TorchScript conversion
- `tensorflow/` - TensorFlow 1.x conversion
- `tensorflow2/` - TensorFlow 2.x and Keras conversion
- `milproto/` - Loading from existing MIL protobuf
- Each frontend parses the source model and builds a MIL Program

**MIL (Model Intermediate Language)** (`coremltools/converters/mil/mil/`):
- Framework-agnostic intermediate representation
- Key data structures:
  - **Program** - Container for MIL functions
  - **Function/Block** - Computation graphs with control flow support
  - **Operation** - Individual computation nodes (SSA form)
  - **Var** - Variables/values with symbolic types
  - **Builder** - Singleton for constructing MIL programs (`mb.add()`, `mb.conv()`, etc.)
- Type system (`mil/types/`) supports:
  - Tensor types with symbolic dimensions (for dynamic shapes)
  - Type inference and compatibility checking
  - Broadcasting rules
- Operations (`mil/ops/defs/`) organized by iOS version (iOS15, iOS16, iOS17, iOS18)

**Optimization Passes** (`coremltools/converters/mil/mil/passes/`):
- **PassPipeline** orchestrates graph transformations
- **GraphPass** base class for all passes
- Common passes in `passes/defs/`:
  - Quantization optimizations
  - Operation fusion (conv, linear, activation, normalization)
  - Dead code elimination and constant propagation
  - Complex dialect lowering
- Passes are ordered to canonicalize, simplify, fuse, and eliminate redundancy

**Backends** (`coremltools/converters/mil/backend/`):
- `mil/` - MLProgram backend (iOS 15+)
  - Creates protobuf messages mapping to MLProgram spec
  - Supports weight file externalization
- `nn/` - Neural Network backend (iOS 14 and earlier)
  - Maps MIL operations to NeuralNetwork specification
  - Large op mapping file (129KB) for translation
- Backend choice depends on `minimum_deployment_target` or `convert_to` parameter

**Protobuf Definitions** (`coremltools/proto/`):
- Python bindings generated from `.proto` files in `mlmodel/format/`
- `MIL_pb2.py` - MLProgram/MIL specification
- `NeuralNetwork_pb2.py` - Neural Network specification (largest, 163KB)
- `Model_pb2.py` - Core ML model container
- `FeatureTypes_pb2.py` - Input/output types

**C++ Bindings**:
- `coremlpython/` - Objective-C/C++ code for macOS model execution
  - Provides `libcoremlpython` module for prediction via CoreML framework
- `milstoragepython/` - C++ utilities for MLPackage serialization
  - Provides `libmilstoragepython` module for weight file handling

**MLModel Class** (`coremltools/models/model.py`):
- High-level wrapper around protobuf or .mlpackage
- Provides API for prediction, metadata management, model inspection
- Integration point for C++ bindings on macOS

**Specialized Converters**:
- `coremltools/converters/sklearn/` - scikit-learn → Core ML (27 converter files)
- `coremltools/converters/xgboost/` - XGBoost → Core ML
- `coremltools/converters/libsvm/` - LibSVM → Core ML
- These bypass MIL and directly create protobuf models

**Optimization** (`coremltools/optimize/`):
- `coreml/` - Post-conversion optimizations (quantization, palettization, pruning)
- `torch/` - Pre-conversion PyTorch optimization

### Architectural Patterns

1. **Visitor Pattern**: Passes traverse and transform graphs via visitors
2. **Registry Pattern**: Operations, passes, and converters use `@register_op`, `@ConverterRegistry` for extensibility
3. **Builder Pattern**: Builder class provides fluent API for MIL program construction
4. **Strategy Pattern**: Optimization passes compose into different pipelines for different targets
5. **Adapter Pattern**: Frontends adapt frameworks to MIL; backends adapt MIL to output formats

### Key Concepts

**Deployment Targets** (`coremltools/converters/mil/_deployment_compatibility.py`):
- Maps iOS/macOS versions to MIL opset versions (CoreML3-CoreML9)
- Determines available operations and features
- Validates model compatibility

**Symbolic Dimensions** (`coremltools/converters/mil/mil/types/symbolic.py`):
- Supports symbolic dimension variables (is0, is1, etc.) for dynamic shapes
- Shapes can be partially known at compile time
- Essential for models with variable batch sizes

**SSA Form**: All MIL operations use Static Single Assignment - each variable assigned once

**Operation Versioning**: Operations are organized by iOS version, ensuring compatibility with target deployment

## Development Workflow

### Adding a New Operation

1. Define operation in `coremltools/converters/mil/mil/ops/defs/iOS<version>/`
2. Use `@register_op` decorator
3. Implement `value_inference()` for type inference
4. Add to backend mappings in `coremltools/converters/mil/backend/*/`
5. Write tests in `coremltools/test/`

### Adding a New Pass

1. Create pass class inheriting from `GraphPass` in `coremltools/converters/mil/mil/passes/defs/`
2. Implement graph transformation logic
3. Register pass in PassRegistry
4. Add to _COMMON_PASSES in `pass_pipeline.py` if it should run by default
5. Write tests validating the transformation

### Adding Frontend Support

1. Create new frontend directory in `coremltools/converters/mil/frontend/`
2. Implement loader that builds MIL Program using Builder
3. Add converter registration via `@ConverterRegistry.frontend`
4. Update `_converters_entry.py` to route to new frontend
5. Add comprehensive tests in frontend's `test/` directory

## Important Files

- `coremltools/__init__.py` - Package entry point, exports main API
- `coremltools/converters/_converters_entry.py` - Main `convert()` function
- `coremltools/converters/mil/mil/builder.py` - MIL program construction
- `coremltools/converters/mil/mil/passes/pass_pipeline.py` - Optimization orchestration
- `coremltools/models/model.py` - MLModel class for model manipulation
- `Makefile` - Build targets and commands
- `setup.py` - Package installation configuration
- `pytest.ini` - Test configuration

## Common Tasks

### Running a Single Test File
```bash
pytest coremltools/test/optimize/torch/quantization/test_quantization.py -v
```

### Running Tests for a Specific Converter
```bash
pytest coremltools/converters/mil/frontend/torch/test/ -v
```

### Debugging Type Inference Issues
- Check `mil/types/` for type system
- Operations must implement `value_inference()` correctly
- Use `Program.validate()` to check type consistency

### Debugging Pass Issues
- Set breakpoints in pass's `apply()` method
- Use `PassPipeline.passes` to see pass order
- Check pass options with `PassOption`

### Building for Specific iOS Version
```python
import coremltools as ct
model = ct.convert(
    pytorch_model,
    minimum_deployment_target=ct.target.iOS16
)
```

### Working with C++ Code
- C++ code is in `coremlpython/`, `milstoragepython/`, `mlmodel/`
- Build C++ changes with `make build`
- Python bindings are generated during build

## Testing Structure

- `coremltools/test/ml_program/` - MIL and MLProgram tests
- `coremltools/test/neural_network/` - NeuralNetwork backend tests
- `coremltools/test/optimize/` - Optimization tests
  - `torch/` - PyTorch optimization tests
  - `coreml/` - Core ML optimization tests
- `coremltools/test/converters/mil/` - MIL infrastructure tests
- Frontend-specific: `coremltools/converters/mil/frontend/*/test/`

## Resources

- [Official Documentation](https://apple.github.io/coremltools/docs-guides/index.html)
- [API Reference](https://apple.github.io/coremltools/index.html)
- [Core ML Specification](https://apple.github.io/coremltools/mlmodel/index.html)
- [Release Notes](https://github.com/apple/coremltools/releases/)
- [Building from Source](BUILDING.md)
- [Contributing Guidelines](CONTRIBUTING.md)
