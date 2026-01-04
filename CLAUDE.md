# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository is a fork of CoreMLTools extended with a specialized KataGo converter. It includes:

1. **CoreMLTools Base**: Python package for converting ML models (TensorFlow, PyTorch, scikit-learn, etc.) to Apple's Core ML format
2. **KataGo Python Converter**: Native Python converter for KataGo neural network models (`coremltools/converters/katago/`)
3. **KataGo C++ Converter**: Standalone C++ library for KataGo conversion without Python dependencies (`katagocoreml/`)

The KataGo converters support model versions 8-16, board sizes 2x2 to 37x37, FLOAT32/FLOAT16 precision, and both standard and human SL (metadata) model variants.

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

### Building CoreMLTools
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

### Building KataGo C++ Converter
```bash
cd katagocoreml
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(sysctl -n hw.ncpu)

# Build targets:
# - libkatagocoreml.a (static library)
# - katago2coreml (CLI tool)
# - katagocoreml_tests (unit tests, if KATAGOCOREML_BUILD_TESTS=ON)
```

**C++ Dependencies**: CMake 3.14+, C++17 compiler, Protobuf, Abseil, zlib

### Testing

#### CoreMLTools General Tests
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

#### KataGo Converter Tests
```bash
# Run KataGo cross-validation tests (requires KataGo executable and model)
pytest coremltools/test/converters/katago/test_cross_validation.py -v

# Run specific test case
pytest coremltools/test/converters/katago/test_cross_validation.py::test_cross_validation_19x19 -v

# Run with verbose output showing tolerances
pytest coremltools/test/converters/katago/test_cross_validation.py -v -s
```

**Test Prerequisites**:
- KataGo model file (`.bin` or `.bin.gz`)
- KataGo C++ executable with Eigen backend (for ground truth)
- Set paths via environment variables or pytest fixtures

#### KataGo C++ Tests
```bash
cd katagocoreml/build
ctest --output-on-failure
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

# Clean KataGo C++ build
rm -rf katagocoreml/build
```

## KataGo Converter Architecture

### Python Converter Pipeline

```
KataGo Binary Model (.bin/.bin.gz)
    |
[PARSER] _katago_parser.py - Parse binary format, extract weights
    |
[TYPES] _katago_types.py - Structured model description (KataGoModelDesc)
    |
[BUILDER] _katago_model_builder.py - Construct MIL program
    |
[OPS] _katago_ops.py - MIL operation builders for KataGo layers
    |
[COREMLTOOLS BACKEND] - Convert MIL to MLProgram protobuf
    |
Core ML Model (.mlpackage)
```

### C++ Converter Pipeline

```
KataGo Binary Model (.bin/.bin.gz)
    |
[PARSER] KataGoParser.cpp - Parse binary format (zlib for gzip)
    |
[TYPES] KataGoTypes.hpp - C++ model description structures
    |
[BUILDER] MILBuilder.cpp + Operations.cpp - Build MIL graph
    |
[SERIALIZER] CoreMLSerializer.cpp - Convert to MLProgram protobuf
    |
[WEIGHTS] WeightSerializer.cpp - Write MILBlob weight files
    |
Core ML Model (.mlpackage)
```

### KataGo Model Components

**Trunk Architecture**:
- Residual blocks (ordinary)
- Global pooling residual blocks (with global features)
- Nested bottleneck residual blocks

**Heads**:
- **Policy Head**: Move probability distribution
- **Value Head**: Win/loss/draw probabilities, score, ownership

**Optional Components**:
- SGF metadata encoder (for human SL models) - 192-channel metadata input

### Key Conversion Options

| Option | Description |
|--------|-------------|
| `board_x_size`, `board_y_size` | Board dimensions (default 19x19) |
| `compute_precision` | FLOAT32 or FLOAT16 |
| `optimize_identity_mask` | Enable mask optimization (~6.5% speedup) |
| `minimum_deployment_target` | iOS/macOS version target |

## High-Level Architecture (CoreMLTools Base)

### Conversion Pipeline

```
Source Model (TF/PyTorch/sklearn/etc)
    |
[FRONTEND] - Loads source model, converts to MIL Program
    |
[MIL Program] - Model Intermediate Language (unified representation)
    |
[OPTIMIZATION PASSES] - Graph optimizations via PassPipeline
    |
[BACKEND] - Converts MIL to target format (neuralnetwork or mlprogram)
    |
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
- `coremltools/converters/sklearn/` - scikit-learn -> Core ML (27 converter files)
- `coremltools/converters/xgboost/` - XGBoost -> Core ML
- `coremltools/converters/libsvm/` - LibSVM -> Core ML
- `coremltools/converters/katago/` - KataGo -> Core ML
- These bypass standard frontends and use specialized parsers

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

### Converting a KataGo Model (Python)
```python
from coremltools.converters.katago import convert

model = convert(
    model_path="kata1-b18c384nbt-s9996604160-d4316597426.bin.gz",
    board_x_size=19,
    board_y_size=19,
    compute_precision=ct.precision.FLOAT16,
    optimize_identity_mask=True
)
model.save("KataGo.mlpackage")
```

### Converting a KataGo Model (C++ CLI)
```bash
./katago2coreml \
    --board-x 19 --board-y 19 \
    --float16 \
    --optimize-identity-mask \
    kata1-b18c384nbt-s9996604160-d4316597426.bin.gz \
    KataGo.mlpackage
```

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

### Modifying KataGo Converter

**Python Changes**:
1. Edit files in `coremltools/converters/katago/`
2. Run cross-validation tests to verify correctness
3. Use `scripts/analyze_tolerances.py` to check error distributions

**C++ Changes**:
1. Edit files in `katagocoreml/src/`
2. Rebuild with `make` in build directory
3. Run `ctest` for unit tests
4. Run cross-validation against Python converter

## Important Files

### KataGo Converter (Python)
- `coremltools/converters/katago/__init__.py` - Public API
- `coremltools/converters/katago/_converter.py` - Main convert() function
- `coremltools/converters/katago/_katago_parser.py` - Binary model parser
- `coremltools/converters/katago/_katago_model_builder.py` - MIL program builder
- `coremltools/converters/katago/_katago_ops.py` - MIL operation builders
- `coremltools/converters/katago/_katago_types.py` - Model type definitions

### KataGo Converter (C++)
- `katagocoreml/include/katagocoreml/KataGoConverter.hpp` - Public API
- `katagocoreml/include/katagocoreml/Options.hpp` - Conversion options
- `katagocoreml/src/parser/KataGoParser.cpp` - Binary model parser
- `katagocoreml/src/builder/MILBuilder.cpp` - MIL program builder
- `katagocoreml/src/builder/Operations.cpp` - MIL operation builders
- `katagocoreml/src/serializer/CoreMLSerializer.cpp` - MLProgram serializer
- `katagocoreml/src/serializer/WeightSerializer.cpp` - Weight file writer
- `katagocoreml/tools/katago2coreml.cpp` - CLI tool

### KataGo Tests & Validation
- `coremltools/test/converters/katago/test_cross_validation.py` - Cross-validation tests
- `coremltools/test/converters/katago/validation_utils.py` - Test utilities
- `coremltools/test/converters/katago/conftest.py` - Pytest fixtures
- `coremltools/test/converters/katago/test_inputs_*/` - Test input JSON files

### KataGo Scripts
- `scripts/generate_test_inputs.py` - Generate test input JSON files
- `scripts/analyze_tolerances.py` - Analyze cross-validation error distributions
- `scripts/benchmark_inference.py` - Measure Core ML inference performance

### CoreMLTools Base
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

### Running KataGo Cross-Validation
```bash
# Set environment variables for test fixtures
export KATAGO_MODEL_PATH=/path/to/model.bin.gz
export KATAGO_EXECUTABLE=/path/to/katago

pytest coremltools/test/converters/katago/test_cross_validation.py -v
```

### Analyzing KataGo Tolerance Errors
```bash
python scripts/analyze_tolerances.py \
    --model /path/to/model.bin.gz \
    --katago-executable /path/to/katago \
    --board-size 19
```

### Benchmarking KataGo Inference
```bash
python scripts/benchmark_inference.py \
    --model KataGo.mlpackage \
    --warmup 10 \
    --iterations 100
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
- CoreMLTools C++ code is in `coremlpython/`, `milstoragepython/`, `mlmodel/`
- KataGo C++ code is in `katagocoreml/`
- Build CoreMLTools C++ changes with `make build`
- Build KataGo C++ changes with `cmake`/`make` in `katagocoreml/build/`

## Testing Structure

- `coremltools/test/ml_program/` - MIL and MLProgram tests
- `coremltools/test/neural_network/` - NeuralNetwork backend tests
- `coremltools/test/optimize/` - Optimization tests
  - `torch/` - PyTorch optimization tests
  - `coreml/` - Core ML optimization tests
- `coremltools/test/converters/mil/` - MIL infrastructure tests
- `coremltools/test/converters/katago/` - KataGo converter tests
- Frontend-specific: `coremltools/converters/mil/frontend/*/test/`
- KataGo C++ tests: `katagocoreml/tests/`

## Resources

### KataGo Converter
- [KataGo Converter Guide](docs/katago/README.md) - Comprehensive setup and usage guide
- [Quick Start](docs/katago/QUICK_START.md) - 5-minute setup
- [C++ Library README](katagocoreml/README.md) - Standalone C++ library documentation
- [KataGo Repository](KataGo/) - KataGo submodule with Eigen backend

### CoreMLTools
- [Official Documentation](https://apple.github.io/coremltools/docs-guides/index.html)
- [API Reference](https://apple.github.io/coremltools/index.html)
- [Core ML Specification](https://apple.github.io/coremltools/mlmodel/index.html)
- [Release Notes](https://github.com/apple/coremltools/releases/)
- [Building from Source](BUILDING.md)
- [Contributing Guidelines](CONTRIBUTING.md)
