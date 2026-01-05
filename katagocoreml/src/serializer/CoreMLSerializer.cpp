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
#include <unordered_map>

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

    // Determine if using FP16 precision
    bool use_fp16 = (options.compute_precision == "FLOAT16");

    // Write weight blob (this sets blob_offset on each WeightEntry)
    writeWeightBlob(weights_dir, weights, use_fp16);

    // Update MIL program with calculated blob offsets
    updateBlobOffsets(program, weights);

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

    // meta_input (optional, for human SL networks with metadata encoder)
    if (options.meta_encoder_version > 0 && options.num_input_meta_channels > 0) {
        auto* meta_input = desc->add_input();
        meta_input->set_name("meta_input");
        auto* meta_type = meta_input->mutable_type()->mutable_multiarraytype();
        meta_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);
        meta_type->add_shape(1);
        meta_type->add_shape(options.num_input_meta_channels);
    }

    // Add output descriptions (names match Python coremltools converter)
    auto* policy_output = desc->add_output();
    policy_output->set_name("policy_p2_conv");
    auto* policy_type = policy_output->mutable_type()->mutable_multiarraytype();
    policy_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* pass_output = desc->add_output();
    // Pass output name: Python uses "policy_pass" for all model versions
    pass_output->set_name("policy_pass");
    auto* pass_type = pass_output->mutable_type()->mutable_multiarraytype();
    pass_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* value_output = desc->add_output();
    value_output->set_name("value_v3_bias");
    auto* value_type = value_output->mutable_type()->mutable_multiarraytype();
    value_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* ownership_output = desc->add_output();
    ownership_output->set_name("value_ownership_conv");
    auto* ownership_type = ownership_output->mutable_type()->mutable_multiarraytype();
    ownership_type->set_datatype(CoreML::Specification::ArrayFeatureType::FLOAT32);

    auto* score_output = desc->add_output();
    score_output->set_name("value_sv3_bias");
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
                                       std::vector<WeightEntry>& weights,
                                       bool use_fp16) {
    std::filesystem::create_directories(weights_dir);
    std::string blob_path = weights_dir + "/weight.bin";
    WeightSerializer::serialize(weights, blob_path, use_fp16);
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

void CoreMLSerializer::updateBlobOffsets(CoreML::Specification::MILSpec::Program* program,
                                          const std::vector<WeightEntry>& weights) {
    // Build a map from weight name to blob offset
    std::unordered_map<std::string, uint64_t> offset_map;
    for (const auto& entry : weights) {
        offset_map[entry.name] = entry.blob_offset;
    }

    // Navigate through MIL program structure to find all blobfilevalue entries
    // Structure: Program -> functions -> blocks -> operations -> attributes["val"]
    for (auto& func_pair : *program->mutable_functions()) {
        auto& func = func_pair.second;
        for (auto& block_pair : *func.mutable_block_specializations()) {
            auto& block = block_pair.second;
            for (int op_idx = 0; op_idx < block.operations_size(); ++op_idx) {
                auto* op = block.mutable_operations(op_idx);
                // Check if this is a const operation
                if (op->type() == "const") {
                    // Get the "val" attribute
                    auto* attrs = op->mutable_attributes();
                    auto val_it = attrs->find("val");
                    if (val_it != attrs->end()) {
                        auto& val = val_it->second;
                        // Check if it's a blobfilevalue
                        if (val.has_blobfilevalue()) {
                            // Get the output name to look up the offset
                            if (op->outputs_size() > 0) {
                                const std::string& output_name = op->outputs(0).name();
                                auto offset_it = offset_map.find(output_name);
                                if (offset_it != offset_map.end()) {
                                    val.mutable_blobfilevalue()->set_offset(offset_it->second);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace katagocoreml
