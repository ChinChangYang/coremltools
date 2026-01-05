// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include "katagocoreml/KataGoConverter.hpp"
#include "parser/KataGoParser.hpp"
#include "builder/MILBuilder.hpp"
#include "serializer/CoreMLSerializer.hpp"
#include <stdexcept>

namespace katagocoreml {

void KataGoConverter::convert(const std::string& input_path,
                               const std::string& output_path,
                               const ConversionOptions& options) {
    // Validate board sizes
    if (options.board_x_size < 2 || options.board_x_size > 37) {
        throw std::invalid_argument("board_x_size must be in range [2, 37]");
    }
    if (options.board_y_size < 2 || options.board_y_size > 37) {
        throw std::invalid_argument("board_y_size must be in range [2, 37]");
    }

    // Parse KataGo model
    KataGoParser parser(input_path);
    KataGoModelDesc model = parser.parse();

    // Determine if using FP16 precision
    bool use_fp16 = (options.compute_precision == "FLOAT16");

    // Build MIL program
    MILBuilder builder(model,
                       options.board_x_size,
                       options.board_y_size,
                       options.optimize_identity_mask,
                       use_fp16);
    auto program = builder.build();

    // Get weights from builder
    auto weights = builder.getWeights();
    std::vector<WeightEntry> weights_copy(weights.begin(), weights.end());

    // Update options with model metadata for serialization
    ConversionOptions final_options = options;
    final_options.model_version = model.model_version;
    final_options.meta_encoder_version = model.meta_encoder_version;
    final_options.num_input_meta_channels = model.num_input_meta_channels;

    // Serialize to .mlpackage
    CoreMLSerializer serializer(final_options.specification_version);
    serializer.serialize(program.get(), weights_copy, output_path, final_options);
}

ModelInfo KataGoConverter::getModelInfo(const std::string& input_path) {
    KataGoParser parser(input_path);
    KataGoModelDesc model = parser.parse();

    ModelInfo info;
    info.name = model.name;
    info.version = model.model_version;
    info.num_input_channels = model.num_input_channels;
    info.num_input_global_channels = model.num_input_global_channels;
    info.num_blocks = model.trunk.num_blocks;
    info.trunk_channels = model.trunk.trunk_num_channels;
    info.has_metadata_encoder = model.meta_encoder_version > 0;
    info.num_policy_channels = model.num_policy_channels;
    info.num_score_value_channels = model.num_score_value_channels;

    return info;
}

}  // namespace katagocoreml
