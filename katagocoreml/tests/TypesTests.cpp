// katagocoreml - Standalone C++ KataGo to Core ML Converter
// Copyright (c) 2025

#include <gtest/gtest.h>
#include "../src/types/KataGoTypes.hpp"
#include "../src/builder/Operations.hpp"

namespace katagocoreml {
namespace test {

TEST(KataGoTypesTest, ConvLayerWeightShape) {
    ConvLayerDesc layer;
    layer.conv_y_size = 3;
    layer.conv_x_size = 3;
    layer.in_channels = 256;
    layer.out_channels = 256;

    auto shape = layer.getWeightShape();
    ASSERT_EQ(shape.size(), 4);
    EXPECT_EQ(shape[0], 256);  // out_channels
    EXPECT_EQ(shape[1], 256);  // in_channels
    EXPECT_EQ(shape[2], 3);    // height
    EXPECT_EQ(shape[3], 3);    // width
}

TEST(KataGoTypesTest, MatMulLayerWeightShape) {
    MatMulLayerDesc layer;
    layer.in_channels = 768;
    layer.out_channels = 256;

    auto shape = layer.getWeightShape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 768);
    EXPECT_EQ(shape[1], 256);
}

TEST(KataGoTypesTest, PolicyChannelsByVersion) {
    EXPECT_EQ(KataGoModelDesc::getPolicyChannels(8), 1);
    EXPECT_EQ(KataGoModelDesc::getPolicyChannels(11), 1);
    EXPECT_EQ(KataGoModelDesc::getPolicyChannels(12), 2);
    EXPECT_EQ(KataGoModelDesc::getPolicyChannels(15), 2);
    EXPECT_EQ(KataGoModelDesc::getPolicyChannels(16), 4);
}

TEST(KataGoTypesTest, ScoreValueChannelsByVersion) {
    EXPECT_EQ(KataGoModelDesc::getScoreValueChannels(8), 4);
    EXPECT_EQ(KataGoModelDesc::getScoreValueChannels(9), 6);
    EXPECT_EQ(KataGoModelDesc::getScoreValueChannels(16), 6);
}

TEST(KataGoTypesTest, MaskConstants19x19) {
    MaskConstants mc(19, 19);
    EXPECT_FLOAT_EQ(mc.mask_sum, 361.0f);
    EXPECT_NEAR(mc.mask_sum_reciprocal, 1.0f / 361.0f, 1e-6f);
    EXPECT_NEAR(mc.mask_sum_sqrt_s14_m01, 0.5f, 0.01f);  // (19 - 14) * 0.1 = 0.5
}

TEST(KataGoTypesTest, MaskConstants9x9) {
    MaskConstants mc(9, 9);
    EXPECT_FLOAT_EQ(mc.mask_sum, 81.0f);
    EXPECT_NEAR(mc.mask_sum_reciprocal, 1.0f / 81.0f, 1e-6f);
    // sqrt(81) = 9, (9 - 14) * 0.1 = -0.5
    EXPECT_NEAR(mc.mask_sum_sqrt_s14_m01, -0.5f, 0.01f);
}

}  // namespace test
}  // namespace katagocoreml
