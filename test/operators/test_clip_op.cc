#include "core/runtime.h"
#include "operators/Clip.h"
#include "gtest/gtest.h"

namespace infini {

class ClipBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// Test basic construction of Clip operator
TEST_F(ClipBasicTest, BasicConstruction) {
    auto X = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto Min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto Max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ClipObj>(X, Min, Max, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Clip);
    EXPECT_EQ(op->getNumInputs(), 3);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

// Test shape inference: output shape = x shape
TEST_F(ClipBasicTest, ShapeInference) {
    auto X = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto Min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto Max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ClipObj>(X, Min, Max, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1);

    auto outputShape = (*inferredShapes)[0];
    EXPECT_TRUE(outputShape->isConcrete());

    auto shapeValues = outputShape->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 3u);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

// Test shape inference with same-shape min/max
TEST_F(ClipBasicTest, ShapeInferenceSameShapeMinMax) {
    auto X = graph->addTensor({5, 6}, DataType(INFINI_DTYPE_F32));
    auto Min = graph->addTensor({5, 6}, DataType(INFINI_DTYPE_F32));
    auto Max = graph->addTensor({5, 6}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ClipObj>(X, Min, Max, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());

    auto outputShape = (*inferredShapes)[0];
    auto shapeValues = outputShape->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2u);
    EXPECT_EQ(shapeValues[0], 5);
    EXPECT_EQ(shapeValues[1], 6);
}

// Test data type inference
TEST_F(ClipBasicTest, DataTypeInference) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto Min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto Max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ClipObj>(X, Min, Max, nullptr);

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

// Test toString output
TEST_F(ClipBasicTest, ToString) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto Min = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto Max = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ClipObj>(X, Min, Max, nullptr);

    std::string str = op->toString();
    EXPECT_NE(str.find("Clip"), std::string::npos);
}

} // namespace infini
