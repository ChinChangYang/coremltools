# KataGo Core ML Backend Integration

Guide for integrating the converted Core ML model as a custom backend in KataGo.

## Overview

This document describes how to integrate the Core ML converted model into KataGo as a native backend, enabling optimized inference on Apple Silicon devices using the Neural Engine.

**Status**: Future work / Proposed implementation

**Benefits:**
- Neural Engine acceleration on Apple Silicon
- Lower power consumption vs GPU
- Native macOS integration
- Optimized memory management

**Trade-offs:**
- macOS/iOS only
- Requires model conversion for each network
- Fixed board size (19×19)

## Current KataGo Backend Architecture

KataGo supports multiple neural network backends:

### Existing Backends

1. **OpenCL** (`cpp/neuralnet/openclbackend.cpp`)
   - GPU acceleration via OpenCL
   - Cross-platform support
   - Mature and widely used

2. **CUDA** (`cpp/neuralnet/cudabackend.cpp`)
   - NVIDIA GPU acceleration
   - High performance on supported hardware

3. **Eigen** (`cpp/neuralnet/eigenbackend.cpp`)
   - CPU-only backend
   - Portable reference implementation

4. **Metal** (`cpp/neuralnet/metalbackend.swift`)
   - Apple GPU acceleration
   - Swift implementation

### Backend Selection

Users specify backend via GTP configuration:

```
# Example config.cfg
nnBackend = opencl
nnModelFile = model.bin.gz
```

## Proposed Core ML Backend

### Design Overview

Add a new backend type that:
1. Loads converted `.mlpackage` models
2. Converts KataGo's internal board representation to Core ML inputs
3. Executes inference via Core ML API
4. Converts Core ML outputs back to KataGo's format

### Architecture

```
KataGo C++ Code
    ↓
CoreMLBackend (C++)
    ↓
CoreMLWrapper (Objective-C++)
    ↓
Core ML Framework (Apple)
    ↓
Neural Engine / GPU / CPU
```

## Implementation Plan

### 1. Core Files to Create/Modify

**New files:**

- `cpp/neuralnet/coremlbackend.h` - Backend interface declaration
- `cpp/neuralnet/coremlbackend.mm` - Backend implementation (Objective-C++)
- `cpp/neuralnet/coremlwrapper.h` - C++ wrapper around Objective-C
- `cpp/neuralnet/coremlwrapper.mm` - Objective-C++ Core ML integration

**Modified files:**

- `cpp/neuralnet/nninterface.cpp` - Register Core ML backend
- `cpp/neuralnet/desc.h` - Add Core ML backend type
- `cpp/CMakeLists.txt` - Add Core ML compilation and linking

### 2. Backend Interface (coremlbackend.h)

```cpp
#ifndef NEURALNET_COREML_BACKEND_H
#define NEURALNET_COREML_BACKEND_H

#include "../neuralnet/nninterface.h"

namespace CoreMLBackend {

class CoreMLBackendImpl : public NeuralNet {
public:
    CoreMLBackendImpl(
        const std::string& modelPath,
        int maxBatchSize,
        int nnXLen,
        int nnYLen
    );
    ~CoreMLBackendImpl();

    // Evaluate positions
    void evaluate(
        const float* inputSpatial,
        const float* inputGlobal,
        const float* inputMask,
        float* outputPolicy,
        float* outputValue,
        float* outputOwnership,
        float* outputScoreValue,
        int batchSize
    );

private:
    void* mlModelWrapper;  // Opaque pointer to Objective-C++ wrapper
    int maxBatchSize;
    int nnXLen;
    int nnYLen;
};

}  // namespace CoreMLBackend

#endif  // NEURALNET_COREML_BACKEND_H
```

### 3. Objective-C++ Wrapper (coremlwrapper.mm)

```objc
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#include "coremlwrapper.h"
#include <vector>

@interface CoreMLWrapper : NSObject {
    MLModel* _model;
}
- (instancetype)initWithModelPath:(NSString*)path;
- (void)predictWithSpatialInput:(const float*)spatialInput
                    globalInput:(const float*)globalInput
                      inputMask:(const float*)inputMask
                   policyOutput:(float*)policyOutput
                    valueOutput:(float*)valueOutput
                ownershipOutput:(float*)ownershipOutput
               scoreValueOutput:(float*)scoreValueOutput
                      batchSize:(int)batchSize
                          nnXLen:(int)nnXLen
                          nnYLen:(int)nnYLen;
@end

@implementation CoreMLWrapper

- (instancetype)initWithModelPath:(NSString*)path {
    self = [super init];
    if (self) {
        NSURL* modelURL = [NSURL fileURLWithPath:path];
        NSError* error = nil;

        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;  // Use Neural Engine + GPU

        _model = [MLModel modelWithContentsOfURL:modelURL
                                   configuration:config
                                           error:&error];

        if (error) {
            NSLog(@"Failed to load Core ML model: %@", error);
            return nil;
        }
    }
    return self;
}

- (void)predictWithSpatialInput:(const float*)spatialInput
                    globalInput:(const float*)globalInput
                      inputMask:(const float*)inputMask
                   policyOutput:(float*)policyOutput
                    valueOutput:(float*)valueOutput
                ownershipOutput:(float*)ownershipOutput
               scoreValueOutput:(float*)scoreValueOutput
                      batchSize:(int)batchSize
                          nnXLen:(int)nnXLen
                          nnYLen:(int)nnYLen {

    @autoreleasepool {
        // Create MLMultiArray inputs
        NSError* error = nil;

        // Spatial input: (1, 22, nnYLen, nnXLen)
        NSArray* spatialShape = @[@(batchSize), @22, @(nnYLen), @(nnXLen)];
        MLMultiArray* spatialArray = [[MLMultiArray alloc]
            initWithShape:spatialShape
            dataType:MLMultiArrayDataTypeFloat32
            error:&error];

        // Copy spatial data
        memcpy(spatialArray.dataPointer, spatialInput,
               batchSize * 22 * nnYLen * nnXLen * sizeof(float));

        // Global input: (1, 19)
        NSArray* globalShape = @[@(batchSize), @19];
        MLMultiArray* globalArray = [[MLMultiArray alloc]
            initWithShape:globalShape
            dataType:MLMultiArrayDataTypeFloat32
            error:&error];

        memcpy(globalArray.dataPointer, globalInput,
               batchSize * 19 * sizeof(float));

        // Input mask: (1, 1, nnYLen, nnXLen)
        NSArray* maskShape = @[@(batchSize), @1, @(nnYLen), @(nnXLen)];
        MLMultiArray* maskArray = [[MLMultiArray alloc]
            initWithShape:maskShape
            dataType:MLMultiArrayDataTypeFloat32
            error:&error];

        memcpy(maskArray.dataPointer, inputMask,
               batchSize * 1 * nnYLen * nnXLen * sizeof(float));

        // Create feature provider
        NSDictionary* inputDict = @{
            @"spatial_input": spatialArray,
            @"global_input": globalArray,
            @"input_mask": maskArray
        };

        MLDictionaryFeatureProvider* input =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:inputDict
                                                               error:&error];

        // Run prediction
        id<MLFeatureProvider> output = [_model predictionFromFeatures:input
                                                                error:&error];

        if (error) {
            NSLog(@"Prediction failed: %@", error);
            return;
        }

        // Extract outputs
        MLMultiArray* policy = [output featureValueForName:@"policy"].multiArrayValue;
        MLMultiArray* value = [output featureValueForName:@"value"].multiArrayValue;
        MLMultiArray* ownership = [output featureValueForName:@"ownership"].multiArrayValue;
        MLMultiArray* scoreValue = [output featureValueForName:@"score_value"].multiArrayValue;

        // Copy to output buffers
        memcpy(policyOutput, policy.dataPointer,
               batchSize * 2 * nnYLen * nnXLen * sizeof(float));
        memcpy(valueOutput, value.dataPointer,
               batchSize * 3 * sizeof(float));
        memcpy(ownershipOutput, ownership.dataPointer,
               batchSize * 1 * nnYLen * nnXLen * sizeof(float));
        memcpy(scoreValueOutput, scoreValue.dataPointer,
               batchSize * 6 * sizeof(float));
    }
}

@end

// C wrapper functions
extern "C" {
    void* CoreMLWrapperCreate(const char* modelPath) {
        NSString* path = [NSString stringWithUTF8String:modelPath];
        CoreMLWrapper* wrapper = [[CoreMLWrapper alloc] initWithModelPath:path];
        return (__bridge_retained void*)wrapper;
    }

    void CoreMLWrapperDestroy(void* wrapper) {
        if (wrapper) {
            CFRelease(wrapper);
        }
    }

    void CoreMLWrapperPredict(
        void* wrapper,
        const float* spatialInput,
        const float* globalInput,
        const float* inputMask,
        float* policyOutput,
        float* valueOutput,
        float* ownershipOutput,
        float* scoreValueOutput,
        int batchSize,
        int nnXLen,
        int nnYLen
    ) {
        CoreMLWrapper* obj = (__bridge CoreMLWrapper*)wrapper;
        [obj predictWithSpatialInput:spatialInput
                         globalInput:globalInput
                           inputMask:inputMask
                        policyOutput:policyOutput
                         valueOutput:valueOutput
                     ownershipOutput:ownershipOutput
                    scoreValueOutput:scoreValueOutput
                           batchSize:batchSize
                              nnXLen:nnXLen
                              nnYLen:nnYLen];
    }
}
```

### 4. C++ Backend Implementation (coremlbackend.mm)

```cpp
#include "coremlbackend.h"
#include "coremlwrapper.h"
#include <stdexcept>

// C wrapper function declarations
extern "C" {
    void* CoreMLWrapperCreate(const char* modelPath);
    void CoreMLWrapperDestroy(void* wrapper);
    void CoreMLWrapperPredict(
        void* wrapper,
        const float* spatialInput,
        const float* globalInput,
        const float* inputMask,
        float* policyOutput,
        float* valueOutput,
        float* ownershipOutput,
        float* scoreValueOutput,
        int batchSize,
        int nnXLen,
        int nnYLen
    );
}

namespace CoreMLBackend {

CoreMLBackendImpl::CoreMLBackendImpl(
    const std::string& modelPath,
    int maxBatchSize,
    int nnXLen,
    int nnYLen
) : maxBatchSize(maxBatchSize), nnXLen(nnXLen), nnYLen(nnYLen) {

    // Verify board size
    if (nnXLen != 19 || nnYLen != 19) {
        throw std::runtime_error(
            "Core ML backend only supports 19x19 board size"
        );
    }

    // Create Core ML wrapper
    mlModelWrapper = CoreMLWrapperCreate(modelPath.c_str());
    if (!mlModelWrapper) {
        throw std::runtime_error(
            "Failed to load Core ML model: " + modelPath
        );
    }
}

CoreMLBackendImpl::~CoreMLBackendImpl() {
    if (mlModelWrapper) {
        CoreMLWrapperDestroy(mlModelWrapper);
    }
}

void CoreMLBackendImpl::evaluate(
    const float* inputSpatial,
    const float* inputGlobal,
    const float* inputMask,
    float* outputPolicy,
    float* outputValue,
    float* outputOwnership,
    float* outputScoreValue,
    int batchSize
) {
    if (batchSize > maxBatchSize) {
        throw std::runtime_error("Batch size exceeds maximum");
    }

    // Note: Current implementation processes one position at a time
    // Future enhancement: support batching
    for (int i = 0; i < batchSize; i++) {
        int spatialOffset = i * 22 * nnYLen * nnXLen;
        int globalOffset = i * 19;
        int maskOffset = i * 1 * nnYLen * nnXLen;
        int policyOffset = i * 2 * nnYLen * nnXLen;
        int valueOffset = i * 3;
        int ownershipOffset = i * 1 * nnYLen * nnXLen;
        int scoreValueOffset = i * 6;

        CoreMLWrapperPredict(
            mlModelWrapper,
            inputSpatial + spatialOffset,
            inputGlobal + globalOffset,
            inputMask + maskOffset,
            outputPolicy + policyOffset,
            outputValue + valueOffset,
            outputOwnership + ownershipOffset,
            outputScoreValue + scoreValueOffset,
            1,  // Process one at a time
            nnXLen,
            nnYLen
        );
    }
}

}  // namespace CoreMLBackend
```

### 5. Register Backend (nninterface.cpp)

```cpp
// In nninterface.cpp

#ifdef USE_COREML_BACKEND
#include "coremlbackend.h"
#endif

// In backend initialization function:
if (backendType == "coreml") {
#ifdef USE_COREML_BACKEND
    return new CoreMLBackend::CoreMLBackendImpl(
        modelPath,
        maxBatchSize,
        nnXLen,
        nnYLen
    );
#else
    throw std::runtime_error("Core ML backend not compiled in");
#endif
}
```

### 6. CMake Configuration (CMakeLists.txt)

```cmake
# Add Core ML backend option
option(USE_COREML_BACKEND "Enable Core ML backend" OFF)

if(USE_COREML_BACKEND)
    if(NOT APPLE)
        message(FATAL_ERROR "Core ML backend requires macOS")
    endif()

    # Find Core ML framework
    find_library(COREML_FRAMEWORK CoreML)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(NOT COREML_FRAMEWORK)
        message(FATAL_ERROR "Core ML framework not found")
    endif()

    # Add Core ML source files
    set(COREML_SOURCES
        cpp/neuralnet/coremlbackend.mm
        cpp/neuralnet/coremlwrapper.mm
    )

    # Set Objective-C++ compiler flags
    set_source_files_properties(
        ${COREML_SOURCES}
        PROPERTIES
        COMPILE_FLAGS "-x objective-c++ -fobjc-arc"
    )

    # Add to target
    target_sources(katago PRIVATE ${COREML_SOURCES})
    target_link_libraries(katago
        ${COREML_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
    )

    target_compile_definitions(katago PRIVATE USE_COREML_BACKEND)
endif()
```

## Configuration

### GTP Config File

```
# config.cfg
nnBackend = coreml
nnModelFile = /path/to/KataGo.mlpackage

# Other settings
maxVisits = 800
numSearchThreads = 6
```

### Building KataGo with Core ML Support

```bash
cd katago
mkdir build
cd build

cmake .. \
    -DUSE_COREML_BACKEND=ON \
    -DCMAKE_OSX_ARCHITECTURES=arm64

make -j8
```

## Testing Strategy

### Unit Tests

```cpp
// test/testcoreml.cpp
#include "coremlbackend.h"
#include <cassert>

void testCoreMLBackend() {
    CoreMLBackend::CoreMLBackendImpl backend(
        "KataGo.mlpackage",
        1,  // batchSize
        19, // nnXLen
        19  // nnYLen
    );

    // Create test inputs
    std::vector<float> spatialInput(1 * 22 * 19 * 19, 0.0f);
    std::vector<float> globalInput(1 * 19, 0.0f);
    std::vector<float> inputMask(1 * 1 * 19 * 19, 1.0f);

    // Allocate outputs
    std::vector<float> policyOutput(1 * 2 * 19 * 19);
    std::vector<float> valueOutput(1 * 3);
    std::vector<float> ownershipOutput(1 * 1 * 19 * 19);
    std::vector<float> scoreValueOutput(1 * 6);

    // Run inference
    backend.evaluate(
        spatialInput.data(),
        globalInput.data(),
        inputMask.data(),
        policyOutput.data(),
        valueOutput.data(),
        ownershipOutput.data(),
        scoreValueOutput.data(),
        1
    );

    // Verify outputs are reasonable
    assert(!std::isnan(valueOutput[0]));
    assert(!std::isinf(valueOutput[0]));

    std::cout << "Core ML backend test passed!" << std::endl;
}
```

### Integration Tests

1. **Compare with Eigen backend**: Verify outputs match within tolerance
2. **Play full games**: Ensure GTP commands work correctly
3. **Benchmark performance**: Measure inference latency
4. **Memory leak testing**: Run extended games and monitor memory

### Performance Benchmarking

```bash
# Benchmark Core ML vs other backends
./katago benchmark -config config_coreml.cfg -model KataGo.mlpackage
./katago benchmark -config config_opencl.cfg -model model.bin.gz

# Expected results on Apple M1:
# Core ML:  5-15ms per inference
# OpenCL:   10-30ms per inference
# Eigen:    100-300ms per inference
```

## Performance Considerations

### Optimization Opportunities

1. **Batch inference**: Process multiple positions simultaneously
2. **Asynchronous prediction**: Use Core ML async APIs for pipelining
3. **Memory pooling**: Reuse MLMultiArray objects
4. **Warm-up**: Run dummy inference at startup for JIT compilation

### Expected Performance

| Hardware | Backend | Latency | Throughput |
|----------|---------|---------|------------|
| M1 | Core ML | 5-15ms | 67-200 inferences/sec |
| M1 | Metal | 10-20ms | 50-100 inferences/sec |
| M1 | OpenCL | 15-30ms | 33-67 inferences/sec |
| M2 | Core ML | 4-12ms | 83-250 inferences/sec |

## Future Enhancements

1. **Variable board sizes**: Support 9×9 and 13×13 boards
2. **Dynamic batching**: Automatically batch requests for efficiency
3. **Model caching**: Cache multiple models for different configurations
4. **iOS support**: Enable running on iPhone/iPad devices

## References

- [Core ML Documentation](https://developer.apple.com/documentation/coreml)
- [KataGo Backend Architecture](https://github.com/lightvector/KataGo/tree/master/cpp/neuralnet)
- [Metal Backend Example](https://github.com/lightvector/KataGo/blob/master/cpp/neuralnet/metalbackend.swift)
- [Core ML Performance Guide](https://developer.apple.com/documentation/coreml/core_ml_api/optimizing_performance)

## Contributing

To contribute to Core ML backend development:

1. Follow KataGo's coding conventions
2. Add comprehensive tests for new features
3. Benchmark performance changes
4. Update this documentation

---

**Last updated**: 2025-12-19
