This branch aims to develop a new feature that allows the user to convert a
KataGo model to a Core ML model in a Python script.

The KataGo model format is defined in KataGo's C++ code. To fully understand the
model format, it is necessary to download the KataGo source code and take it as
a reference.

The KataGo's C++ code is available at https://github.com/lightvector/KataGo.

In the KataGo source code, the KataGo model is loaded by `loadFromFileMaybeGZipped()` in `cpp/neuralnet/desc.cpp`. Then, the function 
creates a `ModelDesc` object, representing the model's specification. The 
`ModelDesc` is defined in `cpp/neuralnet/desc.h`.

To know what operations should be built in the Core ML model, refer to one of the implementation of KataGo's backend, such as `cpp/neuralnet/openclbackend.cpp`, `cpp/neuralnet/eigenbackend.cpp`, or `cpp/neuralnet/metalbackend.cpp`.

For example, in `cpp/neuralnet/openclbackend.cpp`, the function `NeuralNet::getOutput()` is called to get the output of the model. It calls the internal function `gpuHandle->model->apply()` to apply the model to the input. The `gpuHandle->model` is a `Model` object, which is built in the constructor of `ComputeHandle`.

It is much important to understand the `Model` object and its `apply()` function. The `apply()` function first computes mask sum by calling `computeMaskSums()` and then calls `trunk->apply()`, `policyHead->apply()`, and `valueHead->apply()`.

It is also possible to see other backend to understand what operations should be built in the Core ML model. For example, in `cpp/neuralnet/metalbackend.swift`, the function `model.apply()` is called to apply the model to the input. It calls the internal function `graph.run()` to run a Metal Performance Shaders Graph. The `graph` is built in the constructor of `Model`, which creates `Trunk`, `PolicyHead`, and `ValueHead` objects with the `SWModelDesc` object. `SWModelDesc` is a Swift wrapper of `ModelDesc` object, representing the model's specification.

I suggest to download the KataGo source code and write up a detailed document about the model format and the operations to be built in the Core ML model. The document will be used as a reference for the implementation of the feature in the Python script.

The coremltools can be used to convert a PyTorch model to a Core ML model, but it is not suitable for KataGo model because KataGo model is not a PyTorch model. We need to develop a new feature to convert a KataGo model to a Core ML model in a Python script. We may need to modify the coremltools to support KataGo model.

To build the coremltools from source, the user prefers to use a virtual environment (conda) to setup Python 3.11. In this environment, the user can build the coremltools from source and install it in the environment.

It could be helpful to build the coremltools from source with user's previous local changes:
- Makefile:21: Set `python = 3.11`.
- CMakeLists.txt:229: `  set(PLAT_NAME "macosx_${MIN_MAC_OS}_${CMAKE_OSX_ARCHITECTURES}")`
- scripts/build.sh:114: Add `  -DCMAKE_OSX_ARCHITECTURES=arm64 \`

When the coremltools is built in the release mode (wheel), it can be installed in the environment and used to convert a KataGo model to a Core ML model.

A KataGo model is a binary file, which can be downloaded from the KataGo training server. The user can download the model at: https://media.katagotraining.org/uploaded/networks/models/kata1/kata1-b28c512nbt-adam-s11165M-d5387M.bin.gz

I suggest to download the KataGo model and convert it to a Core ML model in the Python script.

To verify the Core ML model, the KataGo's eigen backend can be used to compute the reference output. We ensure the input and output of the Core ML model are the same as the input and output of the KataGo's eigen backend.
