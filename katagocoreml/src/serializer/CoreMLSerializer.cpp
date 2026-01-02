// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include "CoreMLSerializer.hpp"
#include "WeightSerializer.hpp"
#include "MIL.pb.h"
#include "Model.pb.h"
#include "FeatureTypes.pb.h"
#include "ModelPackage.hpp"
#include <fstream>
#include <stdexcept>
#include <filesystem>

namespace katagocoreml {

CoreMLSerializer::CoreMLSerializer(int spec_version)
    : m_spec_version(spec_version) {}

void CoreMLSerializer::serialize(CoreML::Specification::MILSpec::Program* program,
                                 std::vector<WeightEntry>& weights,
                                 const std::string& output_path,
                                 const ConversionOptions& options) {
    // Create temporary directory for weights
    std::filesystem::path temp_dir = std::filesystem::temp_directory_path() / "katagocoreml_weights";
    std::filesystem::create_directories(temp_dir);
    std::string weights_dir = temp_dir.string();

    // Write weight blob
    writeWeightBlob(weights_dir, weights);

    // Create Model spec wrapping the MIL program
    auto model = createModelSpec(program, options);

    // Create .mlpackage
    createPackage(output_path, model.get(), weights_dir);

    // Cleanup temp directory
    std::filesystem::remove_all(temp_dir);
}

std::unique_ptr<CoreML::Specification::Model> CoreMLSerializer::createModelSpec(
    CoreML::Specification::MILSpec::Program* program,
    const ConversionOptions& options) {

    auto model = std::make_unique<CoreML::Specification::Model>();
    model->set_specificationversion(m_spec_version);

    // Set description
    auto* desc = model->mutable_description();

    // Add input descriptions
    // spatial_input
    auto* spatial_input = desc->add_input();
    spatial_input->set_name("spatial_input");
    auto* spatial_type = spatial_input->mutable_type()->mutable_multiarraytype();
    spatial_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    spatial_type->add_shape(1);
    // Note: num_input_channels would need to be passed in
    spatial_type->add_shape(22);  // Default KataGo input channels
    spatial_type->add_shape(options.board_y_size);
    spatial_type->add_shape(options.board_x_size);

    // global_input
    auto* global_input = desc->add_input();
    global_input->set_name("global_input");
    auto* global_type = global_input->mutable_type()->mutable_multiarraytype();
    global_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    global_type->add_shape(1);
    global_type->add_shape(19);  // Default KataGo global channels

    // input_mask
    auto* mask_input = desc->add_input();
    mask_input->set_name("input_mask");
    auto* mask_type = mask_input->mutable_type()->mutable_multiarraytype();
    mask_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
    mask_type->add_shape(1);
    mask_type->add_shape(1);
    mask_type->add_shape(options.board_y_size);
    mask_type->add_shape(options.board_x_size);

    // Add output descriptions
    auto* policy_output = desc->add_output();
    policy_output->set_name("policy_output");
    auto* policy_type = policy_output->mutable_type()->mutable_multiarraytype();
    policy_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* pass_output = desc->add_output();
    pass_output->set_name("pass_output");
    auto* pass_type = pass_output->mutable_type()->mutable_multiarraytype();
    pass_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* value_output = desc->add_output();
    value_output->set_name("value_output");
    auto* value_type = value_output->mutable_type()->mutable_multiarraytype();
    value_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* ownership_output = desc->add_output();
    ownership_output->set_name("ownership_output");
    auto* ownership_type = ownership_output->mutable_type()->mutable_multiarraytype();
    ownership_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* score_output = desc->add_output();
    score_output->set_name("score_value_output");
    auto* score_type = score_output->mutable_type()->mutable_multiarraytype();
    score_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    // Set metadata
    auto* metadata = desc->mutable_metadata();
    metadata->set_shortdescription("KataGo neural network model");
    (*metadata->mutable_userdefined())["board_x_size"] = std::to_string(options.board_x_size);
    (*metadata->mutable_userdefined())["board_y_size"] = std::to_string(options.board_y_size);
    (*metadata->mutable_userdefined())["converter"] = "katagocoreml";
    (*metadata->mutable_userdefined())["converter_version"] = "1.0.0";

    // Set the MIL program (use Swap to transfer ownership)
    auto* ml_program = model->mutable_mlprogram();
    ml_program->Swap(program);

    return model;
}

void CoreMLSerializer::writeWeightBlob(const std::string& weights_dir,
                                       std::vector<WeightEntry>& weights) {
    std::filesystem::create_directories(weights_dir);
    std::string blob_path = weights_dir + "/weight.bin";
    WeightSerializer::serialize(weights, blob_path);
}

void CoreMLSerializer::createPackage(const std::string& output_path,
                                     CoreML::Specification::Model* model,
                                     const std::string& weights_dir) {
    // Create package using MPL::ModelPackage
    MPL::ModelPackage package(output_path, true, false);

    // Serialize model spec to temp file
    std::filesystem::path temp_spec = std::filesystem::temp_directory_path() / "model.mlmodel";
    {
        std::ofstream out(temp_spec, std::ios::binary);
        if (!out) {
            throw std::runtime_error("Failed to create temp model file");
        }
        if (!model->SerializeToOstream(&out)) {
            throw std::runtime_error("Failed to serialize model spec");
        }
    }

    // Set root model
    package.setRootModel(temp_spec.string(), "model.mlmodel", "com.apple.CoreML", "KataGo Core ML Model");

    // Add weights
    package.addItem(weights_dir, "weights", "com.apple.CoreML", "Model Weights");

    // Cleanup temp file
    std::filesystem::remove(temp_spec);
}

}  // namespace katagocoreml
