#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LayerNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LayerNormBasicTest, BasicConstruction) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);

    EXPECT_EQ(op->getOpType(), OpType::LayerNorm);
    EXPECT_EQ(op->getNumInputs(), 3);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(LayerNormBasicTest, ShapeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 2u);
    EXPECT_EQ(shapeVals[0], 2u);
    EXPECT_EQ(shapeVals[1], 4u);
}

TEST_F(LayerNormBasicTest, ShapeInference3D) {
    auto X = graph->addTensor({1, 8, 16}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 3u);
    EXPECT_EQ(shapeVals[0], 1u);
    EXPECT_EQ(shapeVals[1], 8u);
    EXPECT_EQ(shapeVals[2], 16u);
}

TEST_F(LayerNormBasicTest, DataTypeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(LayerNormBasicTest, ToString) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);
    EXPECT_NE(op->toString().find("LayerNorm"), std::string::npos);
}

} // namespace infini
