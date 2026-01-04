// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include "MILBuilder.hpp"
#include <stdexcept>

// Include generated protobuf headers
#include "MIL.pb.h"

namespace katagocoreml {

MILBuilder::MILBuilder(const KataGoModelDesc& model,
                       int board_x_size,
                       int board_y_size,
                       bool optimize_identity_mask)
    : m_model(model)
    , m_board_x_size(board_x_size)
    , m_board_y_size(board_y_size)
    , m_optimize_identity_mask(optimize_identity_mask)
    , m_ops(board_x_size, board_y_size, optimize_identity_mask)
    , m_var_counter(0) {}

std::string MILBuilder::genVarName(const std::string& prefix) {
    return prefix + "_" + std::to_string(m_var_counter++);
}

std::unique_ptr<CoreML::Specification::MILSpec::Program> MILBuilder::build() {
    auto program = std::make_unique<CoreML::Specification::MILSpec::Program>();
    program->set_version(1);

    // Create main function
    auto& functions = *program->mutable_functions();
    auto& main_func = functions["main"];
    main_func.set_opset("CoreML5");

    // Create main block
    auto& blocks = *main_func.mutable_block_specializations();
    auto& main_block = blocks["CoreML5"];

    // Define inputs
    // spatial_input: [1, num_input_ch, board_y, board_x]
    auto* spatial_input = main_func.add_inputs();
    spatial_input->set_name("spatial_input");
    auto* spatial_type = spatial_input->mutable_type()->mutable_tensortype();
    spatial_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    spatial_type->set_rank(4);
    auto* spatial_dim = spatial_type->add_dimensions();
    spatial_dim->mutable_constant()->set_size(1);
    spatial_dim = spatial_type->add_dimensions();
    spatial_dim->mutable_constant()->set_size(m_model.num_input_channels);
    spatial_dim = spatial_type->add_dimensions();
    spatial_dim->mutable_constant()->set_size(m_board_y_size);
    spatial_dim = spatial_type->add_dimensions();
    spatial_dim->mutable_constant()->set_size(m_board_x_size);

    // global_input: [1, num_global_ch]
    auto* global_input = main_func.add_inputs();
    global_input->set_name("global_input");
    auto* global_type = global_input->mutable_type()->mutable_tensortype();
    global_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    global_type->set_rank(2);
    auto* global_dim = global_type->add_dimensions();
    global_dim->mutable_constant()->set_size(1);
    global_dim = global_type->add_dimensions();
    global_dim->mutable_constant()->set_size(m_model.num_input_global_channels);

    // input_mask: [1, 1, board_y, board_x]
    auto* mask_input = main_func.add_inputs();
    mask_input->set_name("input_mask");
    auto* mask_type = mask_input->mutable_type()->mutable_tensortype();
    mask_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    mask_type->set_rank(4);
    auto* mask_dim = mask_type->add_dimensions();
    mask_dim->mutable_constant()->set_size(1);
    mask_dim = mask_type->add_dimensions();
    mask_dim->mutable_constant()->set_size(1);
    mask_dim = mask_type->add_dimensions();
    mask_dim->mutable_constant()->set_size(m_board_y_size);
    mask_dim = mask_type->add_dimensions();
    mask_dim->mutable_constant()->set_size(m_board_x_size);

    // Optional meta_input for human SL networks
    std::string meta_input_name;
    if (m_model.meta_encoder_version > 0 && m_model.num_input_meta_channels > 0) {
        auto* meta_input = main_func.add_inputs();
        meta_input->set_name("meta_input");
        auto* meta_type = meta_input->mutable_type()->mutable_tensortype();
        meta_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        meta_type->set_rank(2);
        auto* meta_dim = meta_type->add_dimensions();
        meta_dim->mutable_constant()->set_size(1);
        meta_dim = meta_type->add_dimensions();
        meta_dim->mutable_constant()->set_size(m_model.num_input_meta_channels);
        meta_input_name = "meta_input";
    }

    // Build the network
    const std::string* meta_ptr = meta_input_name.empty() ? nullptr : &meta_input_name;
    std::string trunk_out = buildTrunk(&main_block, "spatial_input", "global_input", "input_mask", meta_ptr);

    // Build heads
    std::string policy_out, pass_out;
    buildPolicyHead(&main_block, trunk_out, "input_mask", policy_out, pass_out);

    std::string value_out, ownership_out, score_value_out;
    buildValueHead(&main_block, trunk_out, "input_mask", value_out, ownership_out, score_value_out);

    // Set block outputs
    main_block.add_outputs(policy_out);
    main_block.add_outputs(pass_out);
    main_block.add_outputs(value_out);
    main_block.add_outputs(ownership_out);
    main_block.add_outputs(score_value_out);

    return program;
}

// ============================================================================
// MIL Operation Helpers
// ============================================================================

void MILBuilder::addConstOp(CoreML::Specification::MILSpec::Block* block,
                            const std::string& name,
                            const std::vector<float>& data,
                            const std::vector<int64_t>& shape) {
    // Register weight for blob storage
    m_ops.registerWeight(name, data, shape);

    // Add const operation
    auto* op = block->add_operations();
    op->set_type("const");

    // Set output
    auto* output = op->add_outputs();
    output->set_name(name);
    auto* output_type = output->mutable_type()->mutable_tensortype();
    output_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    output_type->set_rank(static_cast<int64_t>(shape.size()));
    for (int64_t dim : shape) {
        auto* d = output_type->add_dimensions();
        d->mutable_constant()->set_size(dim);
    }

    // Set val attribute with blob reference
    auto& attrs = *op->mutable_attributes();
    auto& val_attr = attrs["val"];
    auto* blob_val = val_attr.mutable_blobfilevalue();
    blob_val->set_filename("@model_path/weights/weight.bin");
    // Offset will be set during serialization
}

void MILBuilder::addConvOp(CoreML::Specification::MILSpec::Block* block,
                           const std::string& input,
                           const ConvLayerDesc& layer,
                           const std::string& output) {
    // Create const operations for all parameters (matching Python structure)
    std::string weight_name = output + "_weight_0";
    std::string pad_type_name = output + "_pad_type_0";
    std::string dilations_name = output + "_dilations_0";
    std::string strides_name = output + "_strides_0";
    std::string groups_name = output + "_groups_0";
    std::string pad_name = output + "_pad_0";

    // Add weight constant
    addConstOp(block, weight_name, layer.weights, layer.getWeightShape());

    // Add pad_type constant ("same")
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_immediatevalue()->mutable_tensor()->mutable_strings()->add_values("same");
        auto* out = const_op->add_outputs();
        out->set_name(pad_type_name);
    }

    // Add dilations constant
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(layer.dilation_y);
        int_vals->add_values(layer.dilation_x);
        auto* out = const_op->add_outputs();
        out->set_name(dilations_name);
    }

    // Add strides constant
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(1);
        int_vals->add_values(1);
        auto* out = const_op->add_outputs();
        out->set_name(strides_name);
    }

    // Add groups constant (always 1 for standard convolution)
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
        auto* out = const_op->add_outputs();
        out->set_name(groups_name);
    }

    // Add pad constant [0, 0, 0, 0] for "same" padding
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        auto* int_vals = val.mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        int_vals->add_values(0);
        int_vals->add_values(0);
        int_vals->add_values(0);
        int_vals->add_values(0);
        auto* out = const_op->add_outputs();
        out->set_name(pad_name);
    }

    // Add conv operation referencing all const parameters
    auto* op = block->add_operations();
    op->set_type("conv");

    // Inputs - reference const operations
    auto& inputs = *op->mutable_inputs();
    inputs["dilations"].add_arguments()->set_name(dilations_name);
    inputs["groups"].add_arguments()->set_name(groups_name);
    inputs["pad"].add_arguments()->set_name(pad_name);
    inputs["pad_type"].add_arguments()->set_name(pad_type_name);
    inputs["strides"].add_arguments()->set_name(strides_name);
    inputs["weight"].add_arguments()->set_name(weight_name);
    inputs["x"].add_arguments()->set_name(input);

    // Output
    auto* out = op->add_outputs();
    out->set_name(output);
    auto* out_type = out->mutable_type()->mutable_tensortype();
    out_type->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    out_type->set_rank(4);
}

void MILBuilder::addBatchNormActivationOps(CoreML::Specification::MILSpec::Block* block,
                                           const std::string& input,
                                           const BatchNormLayerDesc& bn,
                                           const ActivationLayerDesc& act,
                                           const std::string& mask,
                                           const std::string& output) {
    // BN: x * scale + bias
    std::string scale_name = output + "_bn_scale";
    std::string bias_name = output + "_bn_bias";

    // Reshape scale/bias to [1, C, 1, 1]
    std::vector<int64_t> bn_shape = {1, static_cast<int64_t>(bn.num_channels), 1, 1};
    addConstOp(block, scale_name, bn.merged_scale, bn_shape);
    addConstOp(block, bias_name, bn.merged_bias, bn_shape);

    // Mul: x * scale
    std::string scaled_name = output + "_scaled";
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        inputs["y"].add_arguments()->set_name(scale_name);
        auto* out = op->add_outputs();
        out->set_name(scaled_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // Add: scaled + bias
    std::string biased_name = output + "_biased";
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(scaled_name);
        inputs["y"].add_arguments()->set_name(bias_name);
        auto* out = op->add_outputs();
        out->set_name(biased_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    std::string bn_output = biased_name;

    // Apply mask if not optimizing
    if (!m_optimize_identity_mask) {
        std::string masked_name = output + "_masked";
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        inputs["y"].add_arguments()->set_name(mask);
        auto* out = op->add_outputs();
        out->set_name(masked_name);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        bn_output = masked_name;
    }

    // Activation
    if (act.activation_type == ActivationType::Identity) {
        // Identity: just rename
        // In MIL we need to copy
        auto* op = block->add_operations();
        op->set_type("identity");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    } else if (act.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bn_output);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    } else if (act.activation_type == ActivationType::Mish) {
        addMishOps(block, bn_output, output);
    }
}

void MILBuilder::addMishOps(CoreML::Specification::MILSpec::Block* block,
                            const std::string& input,
                            const std::string& output) {
    // Mish: x / (1 + 2 / (e * (e + 2)))
    // e = exp(x)

    std::string e = output + "_exp";
    std::string ep2 = output + "_ep2";
    std::string emep2 = output + "_emep2";
    std::string tdemep2 = output + "_tdemep2";
    std::string optdemep2 = output + "_optdemep2";

    // e = exp(x)
    {
        auto* op = block->add_operations();
        op->set_type("exp");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        auto* out = op->add_outputs();
        out->set_name(e);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // ep2 = e + 2
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(e);
        inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(2.0f);
        auto* out = op->add_outputs();
        out->set_name(ep2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // emep2 = e * ep2
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(e);
        inputs["y"].add_arguments()->set_name(ep2);
        auto* out = op->add_outputs();
        out->set_name(emep2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // tdemep2 = 2 / emep2
    {
        auto* op = block->add_operations();
        op->set_type("real_div");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(2.0f);
        inputs["y"].add_arguments()->set_name(emep2);
        auto* out = op->add_outputs();
        out->set_name(tdemep2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // optdemep2 = 1 + tdemep2
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(1.0f);
        inputs["y"].add_arguments()->set_name(tdemep2);
        auto* out = op->add_outputs();
        out->set_name(optdemep2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // output = x / optdemep2
    {
        auto* op = block->add_operations();
        op->set_type("real_div");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(input);
        inputs["y"].add_arguments()->set_name(optdemep2);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }
}

void MILBuilder::addMatMulOp(CoreML::Specification::MILSpec::Block* block,
                             const std::string& input,
                             const MatMulLayerDesc& layer,
                             const std::string& output) {
    // Create const operations for all parameters (matching Python structure)
    std::string weight_name = output + "_y_0";
    std::string transpose_x_name = output + "_transpose_x_0";
    std::string transpose_y_name = output + "_transpose_y_0";

    // Add weight constant
    addConstOp(block, weight_name, layer.weights, layer.getWeightShape());

    // Add transpose_x constant (false)
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(false);
        auto* out = const_op->add_outputs();
        out->set_name(transpose_x_name);
    }

    // Add transpose_y constant (false)
    {
        auto* const_op = block->add_operations();
        const_op->set_type("const");
        auto& val = (*const_op->mutable_attributes())["val"];
        val.mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(false);
        auto* out = const_op->add_outputs();
        out->set_name(transpose_y_name);
    }

    // Add matmul operation
    auto* op = block->add_operations();
    op->set_type("matmul");
    auto& inputs = *op->mutable_inputs();
    inputs["transpose_x"].add_arguments()->set_name(transpose_x_name);
    inputs["transpose_y"].add_arguments()->set_name(transpose_y_name);
    inputs["x"].add_arguments()->set_name(input);
    inputs["y"].add_arguments()->set_name(weight_name);

    auto* out = op->add_outputs();
    out->set_name(output);
    out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
}

void MILBuilder::addMatBiasOp(CoreML::Specification::MILSpec::Block* block,
                              const std::string& input,
                              const MatBiasLayerDesc& layer,
                              const std::string& output) {
    // Add bias constant
    std::string bias_name = output + "_bias";
    std::vector<int64_t> shape = {static_cast<int64_t>(layer.num_channels)};
    addConstOp(block, bias_name, layer.weights, shape);

    // Add add operation
    auto* op = block->add_operations();
    op->set_type("add");
    auto& inputs = *op->mutable_inputs();
    inputs["x"].add_arguments()->set_name(input);
    inputs["y"].add_arguments()->set_name(bias_name);

    auto* out = op->add_outputs();
    out->set_name(output);
    out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
}

void MILBuilder::addGlobalPoolingOps(CoreML::Specification::MILSpec::Block* block,
                                     const std::string& input,
                                     const std::string& mask,
                                     const std::string& output) {
    // KataGo global pooling produces: [mean, mean_scaled, max]
    // mean_scaled = mean * (sqrt(count) - 14) * 0.1

    if (m_optimize_identity_mask) {
        // Optimized path: use precomputed constants
        const auto& mc = m_ops.getMaskConstants();

        // Mean pooling: sum / count
        std::string sum_name = output + "_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(mc.mask_sum_reciprocal);
            auto* out = op->add_outputs();
            out->set_name(mean_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Max pooling
        std::string max_name = output + "_max";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_max");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(max_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean scaled = mean * constant
        std::string mean_scaled_name = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(mc.mask_sum_sqrt_s14_m01);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        std::string mean_flat = output + "_mean_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string max_flat = output + "_max_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(max_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(max_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Concatenate: [mean, mean_scaled, max]
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(max_flat);
            auto& attrs = *op->mutable_attributes();
            attrs["axis"].mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
            auto* out = op->add_outputs();
            out->set_name(output);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }
    } else {
        // Full path with mask operations
        // Count valid positions
        std::string mask_sum_name = output + "_mask_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(mask_sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Masked input
        std::string masked_name = output + "_masked";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["y"].add_arguments()->set_name(mask);
            auto* out = op->add_outputs();
            out->set_name(masked_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Sum masked values
        std::string sum_name = output + "_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean = sum / count
        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("real_div");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mask_sum_name);
            auto* out = op->add_outputs();
            out->set_name(mean_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Max pooling (with mask adjustment)
        std::string mask_minus_one = output + "_mask_m1";
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(1.0f);
            auto* out = op->add_outputs();
            out->set_name(mask_minus_one);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string x_for_max = output + "_x_for_max";
        {
            auto* op = block->add_operations();
            op->set_type("add");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            inputs["y"].add_arguments()->set_name(mask_minus_one);
            auto* out = op->add_outputs();
            out->set_name(x_for_max);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string max_name = output + "_max";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_max");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(x_for_max);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(max_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean scaled = mean * (sqrt(count) - 14) * 0.1
        std::string sqrt_mask = output + "_sqrt_mask";
        {
            auto* op = block->add_operations();
            op->set_type("sqrt");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask_sum_name);
            auto* out = op->add_outputs();
            out->set_name(sqrt_mask);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string sqrt_m14 = output + "_sqrt_m14";
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_mask);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(14.0f);
            auto* out = op->add_outputs();
            out->set_name(sqrt_m14);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string scaled_factor = output + "_scaled_factor";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(0.1f);
            auto* out = op->add_outputs();
            out->set_name(scaled_factor);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(scaled_factor);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Squeeze spatial dimensions
        std::string mean_flat = output + "_mean_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string max_flat = output + "_max_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(max_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(max_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Concatenate: [mean, mean_scaled, max]
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(max_flat);
            auto& attrs = *op->mutable_attributes();
            attrs["axis"].mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
            auto* out = op->add_outputs();
            out->set_name(output);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }
    }
}

void MILBuilder::addGlobalPoolingValueOps(CoreML::Specification::MILSpec::Block* block,
                                          const std::string& input,
                                          const std::string& mask,
                                          const std::string& output) {
    // KataGo value head global pooling produces: [mean, mean_scaled, mean_f3]
    // mean_scaled = mean * (sqrt(count) - 14) * 0.1
    // mean_f3 = mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)

    if (m_optimize_identity_mask) {
        // Optimized path: use precomputed constants
        const auto& mc = m_ops.getMaskConstants();

        // Mean pooling: sum / count
        std::string sum_name = output + "_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(mc.mask_sum_reciprocal);
            auto* out = op->add_outputs();
            out->set_name(mean_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean scaled = mean * constant
        std::string mean_scaled_name = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(mc.mask_sum_sqrt_s14_m01);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean feature 3 = mean * constant
        std::string mean_f3_name = output + "_mean_f3";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(mc.mask_sum_sqrt_s14_m01_sq_s01);
            auto* out = op->add_outputs();
            out->set_name(mean_f3_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Squeeze spatial dimensions: [N, C, 1, 1] -> [N, C]
        std::string mean_flat = output + "_mean_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_f3_flat = output + "_mean_f3_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_f3_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_f3_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Concatenate: [mean, mean_scaled, mean_f3]
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(mean_f3_flat);
            auto& attrs = *op->mutable_attributes();
            attrs["axis"].mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
            auto* out = op->add_outputs();
            out->set_name(output);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }
    } else {
        // Full path with mask operations
        // Count valid positions
        std::string mask_sum_name = output + "_mask_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(mask_sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Masked input
        std::string masked_name = output + "_masked";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(input);
            inputs["y"].add_arguments()->set_name(mask);
            auto* out = op->add_outputs();
            out->set_name(masked_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Sum masked values
        std::string sum_name = output + "_sum";
        {
            auto* op = block->add_operations();
            op->set_type("reduce_sum");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(masked_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            attrs["keep_dims"].mutable_immediatevalue()->mutable_tensor()->mutable_bools()->add_values(true);
            auto* out = op->add_outputs();
            out->set_name(sum_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Mean = sum / count
        std::string mean_name = output + "_mean";
        {
            auto* op = block->add_operations();
            op->set_type("real_div");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sum_name);
            inputs["y"].add_arguments()->set_name(mask_sum_name);
            auto* out = op->add_outputs();
            out->set_name(mean_name);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Compute (sqrt(count) - 14)
        std::string sqrt_mask = output + "_sqrt_mask";
        {
            auto* op = block->add_operations();
            op->set_type("sqrt");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mask_sum_name);
            auto* out = op->add_outputs();
            out->set_name(sqrt_mask);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string sqrt_m14 = output + "_sqrt_m14";
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_mask);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(14.0f);
            auto* out = op->add_outputs();
            out->set_name(sqrt_m14);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Feature 2: Mean * (sqrt(count) - 14) * 0.1
        std::string scaled_factor = output + "_scaled_factor";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(0.1f);
            auto* out = op->add_outputs();
            out->set_name(scaled_factor);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled = output + "_mean_scaled";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(scaled_factor);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Feature 3: Mean * ((sqrt(count) - 14)^2 * 0.01 - 0.1)
        std::string sqrt_m14_sq = output + "_sqrt_m14_sq";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14);
            inputs["y"].add_arguments()->set_name(sqrt_m14);
            auto* out = op->add_outputs();
            out->set_name(sqrt_m14_sq);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string sq_01 = output + "_sq_01";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sqrt_m14_sq);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(0.01f);
            auto* out = op->add_outputs();
            out->set_name(sq_01);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string f3_factor = output + "_f3_factor";
        {
            auto* op = block->add_operations();
            op->set_type("sub");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(sq_01);
            inputs["y"].add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_floats()->add_values(0.1f);
            auto* out = op->add_outputs();
            out->set_name(f3_factor);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_f3 = output + "_mean_f3";
        {
            auto* op = block->add_operations();
            op->set_type("mul");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            inputs["y"].add_arguments()->set_name(f3_factor);
            auto* out = op->add_outputs();
            out->set_name(mean_f3);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Squeeze spatial dimensions
        std::string mean_flat = output + "_mean_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_name);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_scaled_flat = output + "_mean_scaled_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_scaled);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_scaled_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        std::string mean_f3_flat = output + "_mean_f3_flat";
        {
            auto* op = block->add_operations();
            op->set_type("squeeze");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(mean_f3);
            auto& attrs = *op->mutable_attributes();
            auto* axes_val = attrs["axes"].mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            axes_val->add_values(2);
            axes_val->add_values(3);
            auto* out = op->add_outputs();
            out->set_name(mean_f3_flat);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Concatenate: [mean, mean_scaled, mean_f3]
        {
            auto* op = block->add_operations();
            op->set_type("concat");
            auto& inputs = *op->mutable_inputs();
            inputs["values"].add_arguments()->set_name(mean_flat);
            inputs["values"].add_arguments()->set_name(mean_scaled_flat);
            inputs["values"].add_arguments()->set_name(mean_f3_flat);
            auto& attrs = *op->mutable_attributes();
            attrs["axis"].mutable_immediatevalue()->mutable_tensor()->mutable_ints()->add_values(1);
            auto* out = op->add_outputs();
            out->set_name(output);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }
    }
}

// ============================================================================
// Network Component Builders
// ============================================================================

std::string MILBuilder::buildTrunk(CoreML::Specification::MILSpec::Block* block,
                                   const std::string& spatial_input,
                                   const std::string& global_input,
                                   const std::string& mask,
                                   const std::string* meta_input) {
    const auto& trunk = m_model.trunk;

    // Initial conv
    std::string x = genVarName("trunk_init_conv");
    addConvOp(block, spatial_input, trunk.initial_conv, x);

    // Global projection
    std::string global_bias = genVarName("trunk_global_proj");
    addMatMulOp(block, global_input, trunk.initial_matmul, global_bias);

    // Reshape global bias to [1, C, 1, 1]
    std::string global_bias_reshaped = genVarName("trunk_global_reshape");
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(global_bias);
        auto& shape_arg = inputs["shape"];
        auto* shape_val = shape_arg.add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        shape_val->add_values(1);
        shape_val->add_values(-1);
        shape_val->add_values(1);
        shape_val->add_values(1);
        auto* out = op->add_outputs();
        out->set_name(global_bias_reshaped);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // Add global bias
    std::string x_with_global = genVarName("trunk_add_global");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(x);
        inputs["y"].add_arguments()->set_name(global_bias_reshaped);
        auto* out = op->add_outputs();
        out->set_name(x_with_global);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }
    x = x_with_global;

    // Add metadata bias if present
    if (trunk.sgf_metadata_encoder.has_value() && meta_input != nullptr) {
        std::string meta_bias = buildSGFMetadataEncoder(block, *meta_input, *trunk.sgf_metadata_encoder);

        // Reshape meta bias
        std::string meta_bias_reshaped = genVarName("trunk_meta_reshape");
        {
            auto* op = block->add_operations();
            op->set_type("reshape");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(meta_bias);
            auto& shape_arg = inputs["shape"];
            auto* shape_val = shape_arg.add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_ints();
            shape_val->add_values(1);
            shape_val->add_values(-1);
            shape_val->add_values(1);
            shape_val->add_values(1);
            auto* out = op->add_outputs();
            out->set_name(meta_bias_reshaped);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }

        // Add meta bias
        std::string x_with_meta = genVarName("trunk_add_meta");
        {
            auto* op = block->add_operations();
            op->set_type("add");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(x);
            inputs["y"].add_arguments()->set_name(meta_bias_reshaped);
            auto* out = op->add_outputs();
            out->set_name(x_with_meta);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        }
        x = x_with_meta;
    }

    // Apply initial mask
    std::string x_masked = genVarName("trunk_init_mask");
    {
        auto* op = block->add_operations();
        op->set_type("mul");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(x);
        inputs["y"].add_arguments()->set_name(mask);
        auto* out = op->add_outputs();
        out->set_name(x_masked);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }
    x = x_masked;

    // Process residual blocks
    for (size_t i = 0; i < trunk.blocks.size(); i++) {
        const auto& entry = trunk.blocks[i];
        std::string prefix = "trunk_block_" + std::to_string(i);

        if (entry.block_kind == ORDINARY_BLOCK_KIND) {
            const auto& block_desc = std::get<ResidualBlockDesc>(*entry.block);
            x = buildResidualBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == GLOBAL_POOLING_BLOCK_KIND) {
            const auto& block_desc = std::get<GlobalPoolingResidualBlockDesc>(*entry.block);
            x = buildGlobalPoolingResidualBlock(block, x, block_desc, mask, prefix);
        } else if (entry.block_kind == NESTED_BOTTLENECK_BLOCK_KIND) {
            const auto& block_desc = std::get<NestedBottleneckResidualBlockDesc>(*entry.block);
            x = buildNestedBottleneckBlock(block, x, block_desc, mask, prefix);
        }
    }

    // Trunk tip
    std::string trunk_out = genVarName("trunk_tip");
    addBatchNormActivationOps(block, x, trunk.trunk_tip_bn, trunk.trunk_tip_activation, mask, trunk_out);

    return trunk_out;
}

std::string MILBuilder::buildResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                           const std::string& input,
                                           const ResidualBlockDesc& block_desc,
                                           const std::string& mask,
                                           const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // First conv
    std::string conv1_out = genVarName(prefix + "_conv1");
    addConvOp(block, pre_out, block_desc.regular_conv, conv1_out);

    // Mid BN + activation
    std::string mid_out = genVarName(prefix + "_mid");
    addBatchNormActivationOps(block, conv1_out, block_desc.mid_bn, block_desc.mid_activation, mask, mid_out);

    // Second conv
    std::string conv2_out = genVarName(prefix + "_conv2");
    addConvOp(block, mid_out, block_desc.final_conv, conv2_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(conv2_out);
        inputs["y"].add_arguments()->set_name(input);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    return output;
}

std::string MILBuilder::buildGlobalPoolingResidualBlock(CoreML::Specification::MILSpec::Block* block,
                                                         const std::string& input,
                                                         const GlobalPoolingResidualBlockDesc& block_desc,
                                                         const std::string& mask,
                                                         const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // Regular conv
    std::string regular_out = genVarName(prefix + "_regular");
    addConvOp(block, pre_out, block_desc.regular_conv, regular_out);

    // Gpool conv
    std::string gpool_conv_out = genVarName(prefix + "_gpool_conv");
    addConvOp(block, pre_out, block_desc.gpool_conv, gpool_conv_out);

    // Gpool BN + activation
    std::string gpool_bn_out = genVarName(prefix + "_gpool_bn");
    addBatchNormActivationOps(block, gpool_conv_out, block_desc.gpool_bn, block_desc.gpool_activation, mask, gpool_bn_out);

    // Global pooling
    std::string gpool_features = genVarName(prefix + "_gpool_features");
    addGlobalPoolingOps(block, gpool_bn_out, mask, gpool_features);

    // Project to bias
    std::string gpool_bias = genVarName(prefix + "_gpool_bias");
    addMatMulOp(block, gpool_features, block_desc.gpool_to_bias_mul, gpool_bias);

    // Reshape bias
    std::string gpool_bias_reshaped = genVarName(prefix + "_gpool_bias_reshape");
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(gpool_bias);
        auto& shape_arg = inputs["shape"];
        auto* shape_val = shape_arg.add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        shape_val->add_values(1);
        shape_val->add_values(-1);
        shape_val->add_values(1);
        shape_val->add_values(1);
        auto* out = op->add_outputs();
        out->set_name(gpool_bias_reshaped);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // Add bias to regular path
    std::string combined = genVarName(prefix + "_combined");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(regular_out);
        inputs["y"].add_arguments()->set_name(gpool_bias_reshaped);
        auto* out = op->add_outputs();
        out->set_name(combined);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // Mid BN + activation
    std::string mid_out = genVarName(prefix + "_mid");
    addBatchNormActivationOps(block, combined, block_desc.mid_bn, block_desc.mid_activation, mask, mid_out);

    // Final conv
    std::string final_conv_out = genVarName(prefix + "_final");
    addConvOp(block, mid_out, block_desc.final_conv, final_conv_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(final_conv_out);
        inputs["y"].add_arguments()->set_name(input);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    return output;
}

std::string MILBuilder::buildNestedBottleneckBlock(CoreML::Specification::MILSpec::Block* block,
                                                    const std::string& input,
                                                    const NestedBottleneckResidualBlockDesc& block_desc,
                                                    const std::string& mask,
                                                    const std::string& prefix) {
    // Pre BN + activation
    std::string pre_out = genVarName(prefix + "_pre");
    addBatchNormActivationOps(block, input, block_desc.pre_bn, block_desc.pre_activation, mask, pre_out);

    // Pre conv (bottleneck reduction)
    std::string pre_conv_out = genVarName(prefix + "_pre_conv");
    addConvOp(block, pre_out, block_desc.pre_conv, pre_conv_out);

    std::string x = pre_conv_out;

    // Process nested blocks
    for (size_t i = 0; i < block_desc.blocks.size(); i++) {
        const auto& entry = block_desc.blocks[i];
        std::string nested_prefix = prefix + "_nested_" + std::to_string(i);

        if (entry.block_kind == ORDINARY_BLOCK_KIND) {
            const auto& nested = std::get<ResidualBlockDesc>(*entry.block);
            x = buildResidualBlock(block, x, nested, mask, nested_prefix);
        } else if (entry.block_kind == GLOBAL_POOLING_BLOCK_KIND) {
            const auto& nested = std::get<GlobalPoolingResidualBlockDesc>(*entry.block);
            x = buildGlobalPoolingResidualBlock(block, x, nested, mask, nested_prefix);
        }
    }

    // Post BN + activation
    std::string post_out = genVarName(prefix + "_post");
    addBatchNormActivationOps(block, x, block_desc.post_bn, block_desc.post_activation, mask, post_out);

    // Post conv (bottleneck expansion)
    std::string post_conv_out = genVarName(prefix + "_post_conv");
    addConvOp(block, post_out, block_desc.post_conv, post_conv_out);

    // Residual add
    std::string output = genVarName(prefix + "_residual");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(post_conv_out);
        inputs["y"].add_arguments()->set_name(input);
        auto* out = op->add_outputs();
        out->set_name(output);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    return output;
}

void MILBuilder::buildPolicyHead(CoreML::Specification::MILSpec::Block* block,
                                 const std::string& trunk_out,
                                 const std::string& mask,
                                 std::string& policy_out,
                                 std::string& pass_out) {
    const auto& ph = m_model.policy_head;

    // P1 conv
    std::string p1 = genVarName("policy_p1");
    addConvOp(block, trunk_out, ph.p1_conv, p1);

    // G1 conv + BN + activation
    std::string g1_conv = genVarName("policy_g1_conv");
    addConvOp(block, trunk_out, ph.g1_conv, g1_conv);

    std::string g1 = genVarName("policy_g1");
    addBatchNormActivationOps(block, g1_conv, ph.g1_bn, ph.g1_activation, mask, g1);

    // Global pooling on G1
    std::string g1_pooled = genVarName("policy_g1_pool");
    addGlobalPoolingOps(block, g1, mask, g1_pooled);

    // Project to spatial bias
    std::string gpool_bias = genVarName("policy_gpool_bias");
    addMatMulOp(block, g1_pooled, ph.gpool_to_bias_mul, gpool_bias);

    // Reshape bias
    std::string gpool_bias_reshaped = genVarName("policy_gpool_bias_reshape");
    {
        auto* op = block->add_operations();
        op->set_type("reshape");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(gpool_bias);
        auto& shape_arg = inputs["shape"];
        auto* shape_val = shape_arg.add_arguments()->mutable_value()->mutable_immediatevalue()->mutable_tensor()->mutable_ints();
        shape_val->add_values(1);
        shape_val->add_values(-1);
        shape_val->add_values(1);
        shape_val->add_values(1);
        auto* out = op->add_outputs();
        out->set_name(gpool_bias_reshaped);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // Add bias to P1
    std::string p1_biased = genVarName("policy_p1_biased");
    {
        auto* op = block->add_operations();
        op->set_type("add");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(p1);
        inputs["y"].add_arguments()->set_name(gpool_bias_reshaped);
        auto* out = op->add_outputs();
        out->set_name(p1_biased);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    }

    // P1 BN + activation
    std::string p1_activated = genVarName("policy_p1_act");
    addBatchNormActivationOps(block, p1_biased, ph.p1_bn, ph.p1_activation, mask, p1_activated);

    // P2 conv -> policy output
    policy_out = "policy_output";
    addConvOp(block, p1_activated, ph.p2_conv, policy_out);

    // Pass move
    if (ph.gpool_to_pass_mul2.has_value()) {
        // v15+: two-layer pass
        std::string pass_hidden = genVarName("policy_pass_hidden");
        addMatMulOp(block, g1_pooled, ph.gpool_to_pass_mul, pass_hidden);

        std::string pass_biased = genVarName("policy_pass_biased");
        addMatBiasOp(block, pass_hidden, *ph.gpool_to_pass_bias, pass_biased);

        // Activation
        std::string pass_activated = genVarName("policy_pass_act");
        if (ph.pass_activation->activation_type == ActivationType::ReLU) {
            auto* op = block->add_operations();
            op->set_type("relu");
            auto& inputs = *op->mutable_inputs();
            inputs["x"].add_arguments()->set_name(pass_biased);
            auto* out = op->add_outputs();
            out->set_name(pass_activated);
            out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
        } else if (ph.pass_activation->activation_type == ActivationType::Mish) {
            addMishOps(block, pass_biased, pass_activated);
        } else {
            pass_activated = pass_biased;
        }

        pass_out = "pass_output";
        addMatMulOp(block, pass_activated, *ph.gpool_to_pass_mul2, pass_out);
    } else {
        // Pre-v15: single layer pass
        pass_out = "pass_output";
        addMatMulOp(block, g1_pooled, ph.gpool_to_pass_mul, pass_out);
    }
}

void MILBuilder::buildValueHead(CoreML::Specification::MILSpec::Block* block,
                                const std::string& trunk_out,
                                const std::string& mask,
                                std::string& value_out,
                                std::string& ownership_out,
                                std::string& score_value_out) {
    const auto& vh = m_model.value_head;

    // V1 conv + BN + activation
    std::string v1_conv = genVarName("value_v1_conv");
    addConvOp(block, trunk_out, vh.v1_conv, v1_conv);

    std::string v1 = genVarName("value_v1");
    addBatchNormActivationOps(block, v1_conv, vh.v1_bn, vh.v1_activation, mask, v1);

    // Global pooling (value head version)
    std::string v1_pooled = genVarName("value_v1_pool");
    addGlobalPoolingValueOps(block, v1, mask, v1_pooled);

    // V2: linear + bias + activation
    std::string v2_mul = genVarName("value_v2_mul");
    addMatMulOp(block, v1_pooled, vh.v2_mul, v2_mul);

    std::string v2_bias = genVarName("value_v2_bias");
    addMatBiasOp(block, v2_mul, vh.v2_bias, v2_bias);

    std::string v2 = genVarName("value_v2");
    if (vh.v2_activation.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(v2_bias);
        auto* out = op->add_outputs();
        out->set_name(v2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    } else if (vh.v2_activation.activation_type == ActivationType::Mish) {
        addMishOps(block, v2_bias, v2);
    } else {
        v2 = v2_bias;
    }

    // V3: linear + bias -> value output
    std::string v3_mul = genVarName("value_v3_mul");
    addMatMulOp(block, v2, vh.v3_mul, v3_mul);
    value_out = "value_output";
    addMatBiasOp(block, v3_mul, vh.v3_bias, value_out);

    // SV3: score value output
    std::string sv3_mul = genVarName("value_sv3_mul");
    addMatMulOp(block, v2, vh.sv3_mul, sv3_mul);
    score_value_out = "score_value_output";
    addMatBiasOp(block, sv3_mul, vh.sv3_bias, score_value_out);

    // Ownership conv
    ownership_out = "ownership_output";
    addConvOp(block, v1, vh.v_ownership_conv, ownership_out);
}

std::string MILBuilder::buildSGFMetadataEncoder(CoreML::Specification::MILSpec::Block* block,
                                                const std::string& meta_input,
                                                const SGFMetadataEncoderDesc& encoder) {
    // Layer 1
    std::string mul1 = genVarName("meta_mul1");
    addMatMulOp(block, meta_input, encoder.mul1, mul1);

    std::string bias1 = genVarName("meta_bias1");
    addMatBiasOp(block, mul1, encoder.bias1, bias1);

    std::string act1 = genVarName("meta_act1");
    if (encoder.act1.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias1);
        auto* out = op->add_outputs();
        out->set_name(act1);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    } else if (encoder.act1.activation_type == ActivationType::Mish) {
        addMishOps(block, bias1, act1);
    } else {
        act1 = bias1;
    }

    // Layer 2
    std::string mul2 = genVarName("meta_mul2");
    addMatMulOp(block, act1, encoder.mul2, mul2);

    std::string bias2 = genVarName("meta_bias2");
    addMatBiasOp(block, mul2, encoder.bias2, bias2);

    std::string act2 = genVarName("meta_act2");
    if (encoder.act2.activation_type == ActivationType::ReLU) {
        auto* op = block->add_operations();
        op->set_type("relu");
        auto& inputs = *op->mutable_inputs();
        inputs["x"].add_arguments()->set_name(bias2);
        auto* out = op->add_outputs();
        out->set_name(act2);
        out->mutable_type()->mutable_tensortype()->set_datatype(CoreML::Specification::MILSpec::DataType::FLOAT32);
    } else if (encoder.act2.activation_type == ActivationType::Mish) {
        addMishOps(block, bias2, act2);
    } else {
        act2 = bias2;
    }

    // Layer 3 (output)
    std::string output = genVarName("meta_output");
    addMatMulOp(block, act2, encoder.mul3, output);

    return output;
}

}  // namespace katagocoreml
