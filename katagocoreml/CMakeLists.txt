# katagocoreml - Standalone C++ KataGo to Core ML Converter
# Copyright (c) 2025

cmake_minimum_required(VERSION 3.14)
project(katagocoreml VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(KATAGOCOREML_BUILD_TESTS "Build unit tests" ON)
option(KATAGOCOREML_BUILD_TOOLS "Build command-line tools" ON)

# ============================================================================
# External Dependencies
# ============================================================================

find_package(ZLIB REQUIRED)
find_package(Protobuf REQUIRED)
find_package(absl REQUIRED)

# ============================================================================
# Proto Files (compile from source)
# ============================================================================

set(COREMLTOOLS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(PROTO_DIR "${COREMLTOOLS_ROOT}/mlmodel/format")
set(PROTO_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/proto")
file(MAKE_DIRECTORY ${PROTO_GENERATED_DIR})

# Get all proto files
file(GLOB PROTO_FILES "${PROTO_DIR}/*.proto")

# Generate C++ from all proto files
set(PROTO_SRCS)
set(PROTO_HDRS)

foreach(PROTO_FILE ${PROTO_FILES})
    get_filename_component(PROTO_NAME ${PROTO_FILE} NAME_WE)
    set(PROTO_SRC "${PROTO_GENERATED_DIR}/${PROTO_NAME}.pb.cc")
    set(PROTO_HDR "${PROTO_GENERATED_DIR}/${PROTO_NAME}.pb.h")
    list(APPEND PROTO_SRCS ${PROTO_SRC})
    list(APPEND PROTO_HDRS ${PROTO_HDR})

    add_custom_command(
        OUTPUT ${PROTO_SRC} ${PROTO_HDR}
        COMMAND ${Protobuf_PROTOC_EXECUTABLE}
        ARGS --cpp_out=${PROTO_GENERATED_DIR}
             -I${PROTO_DIR}
             ${PROTO_FILE}
        DEPENDS ${PROTO_FILE}
        COMMENT "Generating C++ from ${PROTO_NAME}.proto"
        VERBATIM
    )
endforeach()

# ============================================================================
# MILBlob Sources (vendored from coremltools)
# ============================================================================

set(MILBLOB_DIR "${COREMLTOOLS_ROOT}/mlmodel/src/MILBlob")
set(MILBLOB_SRCS
    "${MILBLOB_DIR}/Blob/FileWriter.cpp"
    "${MILBLOB_DIR}/Blob/StorageWriter.cpp"
    "${MILBLOB_DIR}/Blob/StorageReader.cpp"
    "${MILBLOB_DIR}/Blob/MMapFileReader.cpp"
    "${MILBLOB_DIR}/Blob/MMapFileReaderFactory.cpp"
    "${MILBLOB_DIR}/SubByteTypes.cpp"
    "${MILBLOB_DIR}/Fp8.cpp"
    "${MILBLOB_DIR}/Fp16.cpp"
)

# ============================================================================
# ModelPackage Sources (vendored from coremltools)
# ============================================================================

set(MODELPACKAGE_DIR "${COREMLTOOLS_ROOT}/modelpackage/src")
set(MODELPACKAGE_SRCS
    "${MODELPACKAGE_DIR}/ModelPackage.cpp"
    "${MODELPACKAGE_DIR}/utils/JsonMap.cpp"
)

# ============================================================================
# KataGoCoreML Library Sources
# ============================================================================

set(KATAGOCOREML_SRCS
    src/parser/KataGoParser.cpp
    src/builder/MILBuilder.cpp
    src/builder/Operations.cpp
    src/serializer/CoreMLSerializer.cpp
    src/serializer/WeightSerializer.cpp
    src/Converter.cpp
)

# ============================================================================
# Library Target
# ============================================================================

add_library(katagocoreml STATIC
    ${KATAGOCOREML_SRCS}
    ${PROTO_SRCS}
    ${MILBLOB_SRCS}
    ${MODELPACKAGE_SRCS}
)

target_include_directories(katagocoreml
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${PROTO_GENERATED_DIR}
        ${MILBLOB_DIR}/..
        ${MODELPACKAGE_DIR}
        ${COREMLTOOLS_ROOT}/deps/nlohmann
        ${COREMLTOOLS_ROOT}/deps/FP16/include
)

target_link_libraries(katagocoreml
    PUBLIC
        ZLIB::ZLIB
        protobuf::libprotobuf
        absl::base
        absl::log
        absl::log_internal_check_op
        absl::log_internal_message
        absl::hash
        absl::strings
        absl::status
        absl::statusor
)

# Platform-specific settings
if(APPLE)
    # macOS has built-in UUID support
    target_compile_definitions(katagocoreml PRIVATE APPLE_BUILD=1)
elseif(UNIX)
    find_library(UUID_LIB uuid)
    if(UUID_LIB)
        target_link_libraries(katagocoreml PUBLIC ${UUID_LIB})
    endif()
endif()

# ============================================================================
# Command-Line Tool
# ============================================================================

if(KATAGOCOREML_BUILD_TOOLS)
    add_executable(katago2coreml tools/katago2coreml.cpp)
    target_link_libraries(katago2coreml PRIVATE katagocoreml)
    target_include_directories(katago2coreml PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endif()

# ============================================================================
# Tests
# ============================================================================

if(KATAGOCOREML_BUILD_TESTS)
    enable_testing()

    # Use FetchContent to ensure consistent build configuration
    # (avoid ASAN symbol mismatch with system-installed GTest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_executable(katagocoreml_tests
        tests/ParserTests.cpp
        tests/TypesTests.cpp
    )
    target_link_libraries(katagocoreml_tests
        PRIVATE
            katagocoreml
            GTest::gtest_main
    )
    target_include_directories(katagocoreml_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    include(GoogleTest)
    gtest_discover_tests(katagocoreml_tests)
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS katagocoreml
    EXPORT katagocoreml-targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/katagocoreml
    DESTINATION include
)

if(KATAGOCOREML_BUILD_TOOLS)
    install(TARGETS katago2coreml
        RUNTIME DESTINATION bin
    )
endif()

# ============================================================================
# pkg-config Support
# ============================================================================

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/katagocoreml.pc.in"
    "${CMAKE_CURRENT_BINARY_DIR}/katagocoreml.pc"
    @ONLY
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/katagocoreml.pc"
    DESTINATION lib/pkgconfig
)
