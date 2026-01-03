# katagocoreml

**Standalone C++ library for converting KataGo neural network models to Apple Core ML format**

Pure C++ implementation with no Python dependency - designed for embedding in native applications like KataGo's Core ML backend.

**Quick Links:** [Build](#building-from-source) • [C++ API](#c-api-reference) • [CLI Tool](#cli-tool-usage) • [Integration Examples](#integration-examples)

## Overview

`katagocoreml` is a standalone C++ library that converts KataGo neural network models (.bin/.bin.gz) to Apple's Core ML format (.mlpackage). It's a complete C++ port of the Python-based converter from coremltools, enabling:

- **No Python dependency** - Pure C++17 implementation
- **Embeddable** - Static library for integration into C++ projects
- **Full version support** - KataGo model versions 8-16
- **Flexible board sizes** - 9x9, 13x13, 19x19, or custom dimensions
- **Performance optimization** - Optional `optimize_identity_mask` flag (~6.5% speedup)
- **Precision control** - FLOAT32 or FLOAT16 compute precision

### Use Cases

- **KataGo Core ML backend** - Native Core ML inference without Python overhead
- **macOS/iOS Go applications** - Embed KataGo intelligence in native apps
- **Model conversion pipelines** - Batch convert models for Core ML deployment
- **Research & development** - Experiment with Core ML optimizations

## Quick Start

### Prerequisites Checklist

- ✅ macOS 10.15+ (for Core ML framework)
- ✅ C++17 compiler (Clang 10+ or GCC 9+)
- ✅ CMake 3.14 or later
- ✅ System dependencies: protobuf, abseil, zlib

### Install Dependencies (macOS)

```bash
brew install cmake protobuf abseil zlib
```

### Build in 3 Commands

```bash
cd katagocoreml
mkdir build && cd build
cmake .. && make -j$(nproc)
```

### Test with a KataGo Model

```bash
# Show model info
./katago2coreml --info /path/to/kata1-b40c256.bin.gz

# Convert to Core ML
./katago2coreml /path/to/kata1-b40c256.bin.gz KataGo.mlpackage
```

### Simple C++ API Example

```cpp
#include <katagocoreml/KataGoConverter.hpp>

int main() {
    katagocoreml::KataGoConverter::convert(
        "kata1-b40c256.bin.gz",
        "KataGo.mlpackage"
    );
    return 0;
}
```

## Building from Source

### System Requirements

| Requirement | Minimum Version | Purpose |
|-------------|-----------------|---------|
| **macOS** | 10.15+ | Core ML framework |
| **C++ Compiler** | Clang 10+ / GCC 9+ | C++17 support |
| **CMake** | 3.14+ | Build system |
| **Protobuf** | 3.x | Core ML serialization |
| **Abseil** | Latest | Protobuf dependency |
| **zlib** | System | .gz decompression |

### Installing Dependencies

**macOS (Homebrew):**
```bash
brew install cmake protobuf abseil zlib
```

**Vendored Dependencies** (automatically included):
- MILBlob (from coremltools) - Weight file I/O
- ModelPackage (from coremltools) - .mlpackage creation
- Core ML proto files (from coremltools) - MLProgram specification
- Google Test (auto-fetched) - Testing framework

### Standard Build

```bash
cd katagocoreml
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Build Options

```bash
# Disable tests (faster build)
cmake -DKATAGOCOREML_BUILD_TESTS=OFF ..

# Disable CLI tool (library only)
cmake -DKATAGOCOREML_BUILD_TOOLS=OFF ..

# Custom install prefix
cmake -DCMAKE_INSTALL_PREFIX=/usr/local ..

# Release build with optimizations
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build with symbols
cmake -DCMAKE_BUILD_TYPE=Debug ..
```

### Clean Build

If you encounter build issues (e.g., ASAN symbol mismatches, test linking errors), perform a clean build:

```bash
cd katagocoreml/build
rm -rf CMakeCache.txt CMakeFiles _deps
cmake ..
make -j$(nproc)
```

**Why clean builds matter:**
- CMake caches previous configuration decisions
- Switching between Debug/Release or ASAN on/off requires cleaning
- `_deps/` contains FetchContent downloads (GTest, etc.)
- Clean builds ensure consistent compiler flags across all components

### Running Tests

```bash
# In build directory
ctest --output-on-failure

# Or run tests directly
./katagocoreml_tests

# Verbose test output
./katagocoreml_tests --gtest_filter='*' --gtest_color=yes
```

**Note:** Tests use FetchContent to download Google Test automatically. The first build may take longer while downloading GTest dependencies.

### Installing

```bash
make install
```

**Installs:**
- `lib/libkatagocoreml.a` - Static library
- `include/katagocoreml/*.hpp` - Public headers
- `bin/katago2coreml` - CLI tool

## C++ API Reference

### Basic Usage

```cpp
#include <katagocoreml/KataGoConverter.hpp>

int main() {
    using namespace katagocoreml;

    // Simple conversion with defaults (19x19 board, FLOAT32)
    KataGoConverter::convert(
        "model.bin.gz",
        "KataGo.mlpackage"
    );

    return 0;
}
```

**Compile:**
```bash
# If building in the katagocoreml/build/ directory:
/usr/bin/clang++ -std=c++17 -I../include example.cpp \
    -L. -lkatagocoreml $(pkg-config --libs protobuf) -lz -o example

# Or if katagocoreml is installed (using pkg-config):
/usr/bin/clang++ $(pkg-config --cflags katagocoreml) example.cpp \
    $(pkg-config --libs katagocoreml) -o example
```

### Advanced Usage with Options

```cpp
#include <katagocoreml/KataGoConverter.hpp>
#include <iostream>

int main() {
    using namespace katagocoreml;

    // Configure conversion options
    ConversionOptions opts;
    opts.board_x_size = 19;
    opts.board_y_size = 19;
    opts.optimize_identity_mask = true;  // ~6.5% speedup
    opts.compute_precision = "FLOAT16";   // Smaller model
    opts.specification_version = 6;       // iOS 15+

    try {
        // Get model info first
        auto info = KataGoConverter::getModelInfo("model.bin.gz");
        std::cout << "Model: " << info.name << "\n"
                  << "Version: " << info.version << "\n"
                  << "Blocks: " << info.num_blocks << "\n"
                  << "Channels: " << info.trunk_channels << "\n";

        // Check version support
        if (!KataGoConverter::isVersionSupported(info.version)) {
            std::cerr << "Unsupported version " << info.version << "!\n";
            std::cerr << "Supported: " << KataGoConverter::MIN_SUPPORTED_VERSION
                      << "-" << KataGoConverter::MAX_SUPPORTED_VERSION << "\n";
            return 1;
        }

        // Convert model
        KataGoConverter::convert("model.bin.gz", "output.mlpackage", opts);
        std::cout << "Successfully converted to output.mlpackage\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

### API Classes

#### `KataGoConverter` (Main API)

Static methods for model conversion and inspection.

**Methods:**

```cpp
// Convert KataGo model to Core ML
static void convert(
    const std::string& input_path,   // .bin or .bin.gz file
    const std::string& output_path,  // .mlpackage directory
    const ConversionOptions& options = ConversionOptions{}
);
// Throws: std::runtime_error on conversion failure

// Get model metadata without full conversion
static ModelInfo getModelInfo(const std::string& input_path);
// Throws: std::runtime_error if file cannot be parsed

// Check if a KataGo version is supported
static bool isVersionSupported(int version);

// Get library version
static std::string getVersion();  // Returns "1.0.0"
```

**Constants:**
- `MIN_SUPPORTED_VERSION` = 8
- `MAX_SUPPORTED_VERSION` = 16

#### `ConversionOptions` (Configuration)

```cpp
struct ConversionOptions {
    int board_x_size = 19;               // Board width
    int board_y_size = 19;               // Board height
    bool optimize_identity_mask = false; // Skip mask ops (~6.5% faster)
    std::string compute_precision = "FLOAT32";  // "FLOAT32" or "FLOAT16"
    int specification_version = 6;       // Core ML spec (6 = iOS 15+)
};
```

**Field Descriptions:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `board_x_size` | int | 19 | Board width (9, 13, 19, or custom) |
| `board_y_size` | int | 19 | Board height (9, 13, 19, or custom) |
| `optimize_identity_mask` | bool | false | Skip mask operations for ~6.5% speedup (requires full board) |
| `compute_precision` | string | "FLOAT32" | Compute precision: "FLOAT32" or "FLOAT16" |
| `specification_version` | int | 6 | Core ML spec version (6 = iOS 15+) |

#### `ModelInfo` (Metadata)

```cpp
struct ModelInfo {
    std::string name;                   // Model name from file
    int version;                        // KataGo version (8-16)
    int num_input_channels;             // Spatial input channels (typically 22)
    int num_input_global_channels;      // Global input channels (typically 19)
    int num_blocks;                     // Number of residual blocks
    int trunk_channels;                 // Trunk channel width
    bool has_metadata_encoder;          // Has SGF metadata encoder (human SL)
    int num_policy_channels;            // Policy channels (1, 2, or 4)
    int num_score_value_channels;       // Score value channels (4 or 6)
};
```

## Linking Against katagocoreml

### CMake Integration (Recommended)

If katagocoreml is installed:

```cmake
# In your CMakeLists.txt
find_package(katagocoreml REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE katagocoreml)
```

### Embedding as Subdirectory

If katagocoreml is in your project tree:

```cmake
# Add katagocoreml as subdirectory
add_subdirectory(katagocoreml)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE katagocoreml)
target_include_directories(my_app PRIVATE katagocoreml/include)
```

### Manual Linking

```bash
# Compile and link manually (recommended)
/usr/bin/clang++ -std=c++17 -I/usr/local/include \
    my_app.cpp \
    -L/usr/local/lib -lkatagocoreml \
    $(pkg-config --libs protobuf) -lz \
    -o my_app

# Using pkg-config (if available)
/usr/bin/clang++ -std=c++17 $(pkg-config --cflags katagocoreml) \
    my_app.cpp \
    $(pkg-config --libs katagocoreml) \
    -o my_app
```

## CLI Tool Usage

The `katago2coreml` command-line tool provides a convenient interface for model conversion.

### Basic Usage

```bash
katago2coreml input.bin.gz output.mlpackage
```

### Common Examples

```bash
# Convert standard 19x19 model
katago2coreml kata1-b40c256.bin.gz KataGo.mlpackage

# Convert 9x9 model
katago2coreml -x 9 -y 9 kata1-b6c96-9x9.bin.gz KataGo-9x9.mlpackage

# Enable optimization for ~6.5% speedup
katago2coreml --optimize-identity-mask model.bin.gz KataGo-opt.mlpackage

# Use FLOAT16 for smaller model size
katago2coreml --float16 model.bin.gz KataGo-fp16.mlpackage

# Show model info without converting
katago2coreml --info model.bin.gz

# Verbose output during conversion
katago2coreml -v model.bin.gz output.mlpackage

# Combined options
katago2coreml -x 19 -y 19 --optimize-identity-mask --float16 \
              -v model.bin.gz KataGo-optimized.mlpackage
```

### All CLI Options

```
Usage: katago2coreml [options] <input.bin[.gz]> <output.mlpackage>

Options:
  -x, --board-x <size>     Board width (default: 19)
  -y, --board-y <size>     Board height (default: 19)
  --optimize-identity-mask Optimize for full board (skip mask ops, ~6.5% faster)
  --float16                Use FLOAT16 compute precision (smaller model)
  --info                   Show model info and exit (no conversion)
  -v, --verbose            Enable verbose output
  -h, --help               Show this help message

Examples:
  katago2coreml kata1-b40c256.bin.gz KataGo.mlpackage
  katago2coreml -x 9 -y 9 model.bin.gz KataGo-9x9.mlpackage
  katago2coreml --optimize-identity-mask model.bin.gz KataGo-opt.mlpackage
  katago2coreml --info model.bin.gz

Supported KataGo model versions: 8-16
```

## Integration Examples

### Example 1: KataGo Core ML Backend

Integrate into KataGo's neural network loading code:

```cpp
// In KataGo's neuralnet/nninterface.cpp or similar
#ifdef USE_COREML_BACKEND

#include <katagocoreml/KataGoConverter.hpp>
#include <filesystem>

namespace {

std::string convertToCoreML(const std::string& modelPath, int boardSize) {
    using namespace katagocoreml;

    // Configure conversion
    ConversionOptions opts;
    opts.board_x_size = boardSize;
    opts.board_y_size = boardSize;
    opts.optimize_identity_mask = true;  // KataGo always uses full board
    opts.compute_precision = "FLOAT32";   // Full precision for analysis

    // Create temp path for .mlpackage
    std::string mlpackage_path = std::filesystem::temp_directory_path() /
                                 "katago_coreml.mlpackage";

    // Check if already converted
    if (!std::filesystem::exists(mlpackage_path)) {
        std::cout << "Converting KataGo model to Core ML..." << std::endl;
        KataGoConverter::convert(modelPath, mlpackage_path, opts);
        std::cout << "Conversion complete." << std::endl;
    }

    return mlpackage_path;
}

}  // namespace

void NeuralNet::loadModel(const std::string& modelPath, int boardSize) {
    // Convert to Core ML
    std::string coreml_path = convertToCoreML(modelPath, boardSize);

    // Load with Core ML framework (Objective-C++ code)
    loadCoreMLModel(coreml_path);
}

#endif  // USE_COREML_BACKEND
```

### Example 2: Model Validation Tool

Build a tool to validate KataGo models before deployment:

```cpp
#include <katagocoreml/KataGoConverter.hpp>
#include <iostream>
#include <filesystem>
#include <vector>

void validateModel(const std::string& path) {
    using namespace katagocoreml;

    std::cout << "Checking: " << path << "\n";

    // Check file exists
    if (!std::filesystem::exists(path)) {
        std::cerr << "  ✗ File not found\n";
        return;
    }

    try {
        // Parse model
        auto info = KataGoConverter::getModelInfo(path);

        // Check version support
        if (KataGoConverter::isVersionSupported(info.version)) {
            std::cout << "  ✓ Valid KataGo model (version " << info.version << ")\n";
        } else {
            std::cout << "  ✗ Unsupported version " << info.version << "\n";
            std::cout << "    Supported: " << KataGoConverter::MIN_SUPPORTED_VERSION
                      << "-" << KataGoConverter::MAX_SUPPORTED_VERSION << "\n";
            return;
        }

        // Print architecture
        std::cout << "  Architecture: " << info.num_blocks << "b"
                  << info.trunk_channels << "c\n";
        std::cout << "  Policy channels: " << info.num_policy_channels << "\n";
        std::cout << "  Score value channels: " << info.num_score_value_channels << "\n";

        if (info.has_metadata_encoder) {
            std::cout << "  Note: Has SGF metadata encoder (human SL network)\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "  ✗ Invalid model: " << e.what() << "\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin[.gz]> ...\n";
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        validateModel(argv[i]);
    }

    return 0;
}
```

### Example 3: Batch Converter

Convert multiple models with different configurations:

```cpp
#include <katagocoreml/KataGoConverter.hpp>
#include <iostream>
#include <vector>
#include <string>

struct ConversionJob {
    std::string input;
    std::string output;
    int board_size;
    bool optimize;
};

void batchConvert(const std::vector<ConversionJob>& jobs) {
    using namespace katagocoreml;

    for (const auto& job : jobs) {
        std::cout << "Converting: " << job.input << " -> " << job.output << "\n";

        ConversionOptions opts;
        opts.board_x_size = job.board_size;
        opts.board_y_size = job.board_size;
        opts.optimize_identity_mask = job.optimize;

        try {
            KataGoConverter::convert(job.input, job.output, opts);
            std::cout << "  ✓ Success\n";
        } catch (const std::exception& e) {
            std::cerr << "  ✗ Error: " << e.what() << "\n";
        }
    }
}

int main() {
    std::vector<ConversionJob> jobs = {
        {"kata1-b40c256.bin.gz", "KataGo-19x19.mlpackage", 19, true},
        {"kata1-b6c96-9x9.bin.gz", "KataGo-9x9.mlpackage", 9, true},
        {"kata1-b10c128-13x13.bin.gz", "KataGo-13x13.mlpackage", 13, true},
    };

    batchConvert(jobs);
    return 0;
}
```

## Performance Considerations

### optimize_identity_mask Flag

**What it does:**
- Skips mask-related operations in the MIL program
- Reduces computation by ~6.5% during Core ML inference
- Replaces dynamic mask operations with precomputed constants

**When to use:**
- ✅ Full board inference (all positions valid) - typical for KataGo
- ✅ Production deployments where performance matters
- ✅ Models used for self-play or analysis

**When NOT to use:**
- ❌ If you need to mask out illegal moves at conversion time
- ❌ If board positions vary dynamically

**Example:**
```cpp
ConversionOptions opts;
opts.optimize_identity_mask = true;  // ~6.5% faster inference
KataGoConverter::convert("model.bin.gz", "output.mlpackage", opts);
```

### FLOAT16 vs FLOAT32

| Aspect | FLOAT32 | FLOAT16 |
|--------|---------|---------|
| **Model size** | Baseline | ~50% smaller |
| **Inference speed** | Baseline | Slightly faster on Apple Silicon |
| **Precision** | Full precision | Reduced precision |
| **Use case** | Analysis, match review | Mobile, size-constrained |

**Recommendation:**
- Use **FLOAT32** for analysis and match review (full precision)
- Use **FLOAT16** for mobile apps or when storage is limited

**Example:**
```cpp
ConversionOptions opts;
opts.compute_precision = "FLOAT16";  // Smaller model
KataGoConverter::convert("model.bin.gz", "output.mlpackage", opts);
```

### Board Size Impact

Model size scales with board dimensions:

| Board Size | Relative Size | Use Case |
|------------|---------------|----------|
| 9x9 | ~40% of 19x19 | Beginner training, fast analysis |
| 13x13 | ~60% of 19x19 | Intermediate play |
| 19x19 | 100% (baseline) | Standard Go |

## Troubleshooting

### Build Issues

#### Problem: `CMake Error: Could NOT find Protobuf`

**Solution:**
```bash
# macOS
brew install protobuf

# Ubuntu/Debian
sudo apt-get install libprotobuf-dev protobuf-compiler
```

#### Problem: `CMake Error: Could NOT find absl`

**Solution:**
```bash
# macOS
brew install abseil

# Ubuntu/Debian
sudo apt-get install libabsl-dev
```

#### Problem: `CMake can't find proto files`

**Error message:**
```
CMake Error: Cannot find source file: mlmodel/format/MIL.proto
```

**Solution:**
Ensure you're building from within the coremltools repository. The katagocoreml library expects proto files at `../mlmodel/format/`:

```bash
# Correct location
cd /path/to/KataGoCoremltools/katagocoreml
mkdir build && cd build
cmake ..
```

#### Problem: `undefined reference to google::protobuf::...`

**Solution:**
Link against protobuf and its dependencies:
```bash
clang++ ... -lprotobuf -labsl_base -labsl_strings
```

#### Problem: Test build fails with ASAN symbol errors

**Error message:**
```
Undefined symbols for architecture arm64:
  "___asan_init", referenced from:
  "___asan_stack_malloc_0", referenced from:
  "___asan_handle_no_return", referenced from:
  ...
ld: symbol(s) not found for architecture arm64
```

**Cause:**
System-installed Google Test (e.g., from Homebrew) was built with different compiler flags (ASAN) than katagocoreml, causing symbol mismatches during linking.

**Solution:**
Perform a clean build to force FetchContent to download and build GTest with consistent flags:
```bash
cd katagocoreml/build
rm -rf CMakeCache.txt CMakeFiles _deps
cmake ..
make -j$(nproc)
```

The build system now uses FetchContent by default (not `find_package(GTest)`) to ensure all components are built with the same compiler flags.

### Conversion Issues

#### Problem: "Unsupported model version"

**Error message:**
```
Error: Unsupported KataGo model version: 17
```

**Solution:**
- Check model version: `katago2coreml --info model.bin.gz`
- Supported versions: 8-16
- Use a compatible KataGo model or update the library

#### Problem: Segmentation fault during conversion

**Diagnosis:**
Build with Address Sanitizer:
```bash
cd build
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -g" ..
make
./katago2coreml model.bin.gz output.mlpackage
```

**Common causes:**
- Corrupted model file - verify with `file model.bin.gz`
- Incompatible model version - check with `--info`
- Out of memory - monitor with Activity Monitor

#### Problem: Output .mlpackage doesn't work

**Validation steps:**

1. Verify Core ML framework version:
```bash
sw_vers  # Should be macOS 12+ (Monterey or later)
```

2. Validate .mlpackage structure:
```bash
xcrun coremlcompiler compile output.mlpackage
```

3. Check board size matches inputs:
```cpp
// Ensure conversion board size matches runtime inputs
opts.board_x_size = 19;  // Must match your input tensors
opts.board_y_size = 19;
```

#### Problem: "Failed to decompress .bin.gz file"

**Solution:**
- Verify file integrity: `gzip -t model.bin.gz`
- Re-download the model if corrupted
- Try uncompressed .bin file directly

## Technical Details

### Architecture

```
KataGo Binary (.bin.gz)
    ↓ zlib decompression
[KataGoParser]
    ↓ Binary parsing, weight extraction
[ModelDesc] - In-memory representation
    ↓ MIL operation construction
[MILBuilder]
    ↓ MIL protobuf program
[CoreMLSerializer]
    ↓ Weight blob + .mlpackage creation
Core ML Model (.mlpackage)
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Parser** | `parser/KataGoParser.cpp` | Binary format parsing, zlib decompression |
| **Types** | `types/KataGoTypes.hpp` | Model structure definitions (port of Python) |
| **Builder** | `builder/MILBuilder.cpp` | MIL (Model Intermediate Language) program construction |
| **Operations** | `builder/Operations.cpp` | Weight management, mask constants |
| **Serializer** | `serializer/CoreMLSerializer.cpp` | Protobuf serialization to .mlpackage |
| **Weight Writer** | `serializer/WeightSerializer.cpp` | Blob file creation |

### Vendored Code

This library includes code from coremltools (Apache 2.0 license):

| Component | Source | Purpose |
|-----------|--------|---------|
| **MILBlob** | `mlmodel/src/MILBlob/` | Weight file I/O (StorageWriter, StorageReader) |
| **ModelPackage** | `modelpackage/src/` | .mlpackage directory structure creation |
| **Proto Files** | `mlmodel/format/` | Core ML protobuf definitions (MIL.proto, Model.proto) |

### Conversion Process

1. **Parse** - Decompress .gz, read binary format, extract weights
2. **Build** - Construct MIL operations (conv, bn, matmul, pooling, etc.)
3. **Optimize** - Apply identity mask optimization if enabled
4. **Serialize** - Write protobuf + weight blob to .mlpackage

### MIL Operations Used

Core operations in the generated MLProgram:

- **conv** - 2D convolution (SAME padding)
- **batch_norm** - Batch normalization
- **add/mul** - Element-wise arithmetic
- **relu/mish** - Activation functions (mish = custom 6-op formula)
- **matmul** - Matrix multiplication (fully connected layers)
- **reduce_mean/max** - Global pooling
- **concat** - Channel concatenation
- **reshape/squeeze** - Tensor shape manipulation

## Version Compatibility

### KataGo Model Versions

| KataGo Version | Support Status | Policy Channels | Score Value Channels | Notes |
|----------------|----------------|-----------------|----------------------|-------|
| 8 | ✅ Supported | 1 | 4 | Original architecture |
| 9-11 | ✅ Supported | 1 | 6 | Extended score value |
| 12-15 | ✅ Supported | 2 | 6 | Dual policy channels |
| 16 | ✅ Supported | 4 | 6 | Quad policy channels |
| 17+ | ❌ Not supported | - | - | Requires library update |

### Core ML Versions

| Specification Version | iOS Version | macOS Version | Features |
|-----------------------|-------------|---------------|----------|
| 6 (default) | iOS 15+ | macOS 12+ | MLProgram with MIL |
| 7 | iOS 16+ | macOS 13+ | Additional ops |
| 8 | iOS 17+ | macOS 14+ | Latest features |

**Recommendation:** Use version 6 (default) for maximum compatibility.

## Contributing & Development

### Building Tests

```bash
cd katagocoreml/build
make katagocoreml_tests
./katagocoreml_tests --gtest_color=yes
```

**Test coverage:**
- Parser unit tests (`tests/ParserTests.cpp`)
- Type system tests (`tests/TypesTests.cpp`)

### Adding Support for New KataGo Versions

1. **Update version mappings** in `src/types/KataGoTypes.hpp`:
   ```cpp
   static int getPolicyChannels(int version) {
       if (version <= 11) return 1;
       if (version <= 15) return 2;
       if (version <= 17) return 4;  // Add new version
       return 8;  // Future default
   }
   ```

2. **Add version-specific logic** in `src/builder/MILBuilder.cpp` if architecture changes

3. **Update constants** in `include/katagocoreml/KataGoConverter.hpp`:
   ```cpp
   static constexpr int MAX_SUPPORTED_VERSION = 17;  // Update
   ```

4. **Add tests** in `tests/TypesTests.cpp`:
   ```cpp
   TEST(KataGoTypesTest, PolicyChannelsByVersion) {
       EXPECT_EQ(KataGoModelDesc::getPolicyChannels(17), 4);
   }
   ```

### Code Structure

```
katagocoreml/
├── include/katagocoreml/     # Public API (install these)
│   ├── KataGoConverter.hpp   # Main API class
│   └── Options.hpp           # Configuration structs
├── src/                      # Implementation (private)
│   ├── parser/               # Binary format parsing
│   ├── types/                # Model structures
│   ├── builder/              # MIL construction
│   ├── serializer/           # Protobuf output
│   └── Converter.cpp         # API implementation
├── tests/                    # Google Test unit tests
│   ├── ParserTests.cpp
│   └── TypesTests.cpp
└── tools/                    # CLI tool
    └── katago2coreml.cpp     # Command-line interface
```

### Debugging Tips

**Enable verbose logging:**
```bash
# Use -v flag with CLI tool
katago2coreml -v model.bin.gz output.mlpackage

# Or set environment variable
export KATAGOCOREML_VERBOSE=1
```

**Debug with LLDB:**
```bash
lldb ./katago2coreml
(lldb) breakpoint set -n katagocoreml::KataGoConverter::convert
(lldb) run model.bin.gz output.mlpackage
```

**Memory debugging:**
```bash
cmake -DCMAKE_CXX_FLAGS="-fsanitize=address -g" ..
make
./katagocoreml_tests
```

## License & Credits

```
katagocoreml - Standalone C++ KataGo to Core ML Converter
Copyright (c) 2025

Licensed under the Apache License 2.0 (same as coremltools)

Uses code from:
- coremltools (Apache 2.0) - MILBlob, ModelPackage, proto files
  https://github.com/apple/coremltools
- Google Test (BSD-3-Clause) - Testing framework (dev dependency)
  https://github.com/google/googletest
```

**Acknowledgments:**
- KataGo project by lightvector for the amazing Go engine
- Apple coremltools team for the Python converter this is ported from
- Core ML team at Apple for the MLProgram framework

## References

- **[KataGo](https://github.com/lightvector/KataGo)** - Original Go engine and model format
- **[Core ML Documentation](https://developer.apple.com/documentation/coreml)** - Apple's ML framework
- **[coremltools](https://github.com/apple/coremltools)** - Python converter (source of this port)
- **[Core ML Protobuf Specification](https://github.com/apple/coremltools/tree/main/mlmodel/format)** - MLProgram schema
- **[MIL Operations Reference](https://apple.github.io/coremltools/mlmodel/Format/MIL.html)** - Available MIL ops

## Support

For issues, questions, or contributions:
- **Issues:** Report bugs or feature requests on GitHub
- **Discussions:** Technical questions and architecture discussions
- **Pull Requests:** Contributions welcome (follow existing code style)

---

**katagocoreml** v1.0.0 • Pure C++ KataGo to Core ML Converter • Apache 2.0 License
