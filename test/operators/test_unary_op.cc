#include "core/runtime.h"
#include "operators/Unary.h"
#include "gtest/gtest.h"

namespace infini {

class UnaryBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// ---- Relu ----

TEST_F(UnaryBasicTest, BasicConstruction_Relu) {
    auto X = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Relu, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Relu);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Relu) {
    auto X = graph->addTensor({2, 3, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Relu, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    ASSERT_EQ(inferredShapes->size(), 1u);

    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 3u);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
    EXPECT_EQ(shapeValues[2], 4);
}

TEST_F(UnaryBasicTest, DataTypeInference_Relu) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Relu, X, nullptr);

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Relu) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Relu, X, nullptr);
    std::string str = op->toString();
    EXPECT_NE(str.find("Relu"), std::string::npos);
}

// ---- Sigmoid ----

TEST_F(UnaryBasicTest, BasicConstruction_Sigmoid) {
    auto X = graph->addTensor({4, 5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Sigmoid, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Sigmoid);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Sigmoid) {
    auto X = graph->addTensor({4, 5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Sigmoid, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2u);
    EXPECT_EQ(shapeValues[0], 4);
    EXPECT_EQ(shapeValues[1], 5);
}

TEST_F(UnaryBasicTest, DataTypeInference_Sigmoid) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Sigmoid, X, nullptr);
    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Sigmoid) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Sigmoid, X, nullptr);
    EXPECT_NE(op->toString().find("Sigmoid"), std::string::npos);
}

// ---- Silu ----

TEST_F(UnaryBasicTest, BasicConstruction_Silu) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Silu, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Silu);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Silu) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Silu, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2u);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 4);
}

TEST_F(UnaryBasicTest, DataTypeInference_Silu) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Silu, X, nullptr);
    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Silu) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Silu, X, nullptr);
    EXPECT_NE(op->toString().find("Silu"), std::string::npos);
}

// ---- Gelu ----

TEST_F(UnaryBasicTest, BasicConstruction_Gelu) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Gelu, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Gelu);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Gelu) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Gelu, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2u);
    EXPECT_EQ(shapeValues[0], 3);
    EXPECT_EQ(shapeValues[1], 3);
}

TEST_F(UnaryBasicTest, DataTypeInference_Gelu) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Gelu, X, nullptr);
    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Gelu) {
    auto X = graph->addTensor({3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Gelu, X, nullptr);
    EXPECT_NE(op->toString().find("Gelu"), std::string::npos);
}

// ---- Softplus ----

TEST_F(UnaryBasicTest, BasicConstruction_Softplus) {
    auto X = graph->addTensor({5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Softplus, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Softplus);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Softplus) {
    auto X = graph->addTensor({5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Softplus, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 1u);
    EXPECT_EQ(shapeValues[0], 5);
}

TEST_F(UnaryBasicTest, DataTypeInference_Softplus) {
    auto X = graph->addTensor({5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Softplus, X, nullptr);
    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Softplus) {
    auto X = graph->addTensor({5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Softplus, X, nullptr);
    EXPECT_NE(op->toString().find("Softplus"), std::string::npos);
}

// ---- Tanh ----

TEST_F(UnaryBasicTest, BasicConstruction_Tanh) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Tanh, X, nullptr);

    EXPECT_EQ(op->getOpType(), OpType::Tanh);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(UnaryBasicTest, ShapeInference_Tanh) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Tanh, X, nullptr);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeValues = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeValues.size(), 2u);
    EXPECT_EQ(shapeValues[0], 2);
    EXPECT_EQ(shapeValues[1], 3);
}

TEST_F(UnaryBasicTest, DataTypeInference_Tanh) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Tanh, X, nullptr);
    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(UnaryBasicTest, ToString_Tanh) {
    auto X = graph->addTensor({2, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<UnaryObj>(OpType::Tanh, X, nullptr);
    EXPECT_NE(op->toString().find("Tanh"), std::string::npos);
}

} // namespace infini
