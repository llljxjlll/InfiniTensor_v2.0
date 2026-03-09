#include "core/runtime.h"
#include "operators/Softmax.h"
#include "gtest/gtest.h"

namespace infini {

class SoftmaxBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

// ---- Softmax tests ----

TEST_F(SoftmaxBasicTest, BasicConstruction) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::Softmax, X, nullptr, -1);

    EXPECT_EQ(op->getOpType(), OpType::Softmax);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(SoftmaxBasicTest, ShapeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::Softmax, X, nullptr, -1);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 2u);
    EXPECT_EQ(shapeVals[0], 2u);
    EXPECT_EQ(shapeVals[1], 4u);
}

TEST_F(SoftmaxBasicTest, DataTypeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::Softmax, X, nullptr, -1);

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(SoftmaxBasicTest, AxisStored) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::Softmax, X, nullptr, 1);
    EXPECT_EQ(op->getAxis(), 1);
}

TEST_F(SoftmaxBasicTest, ToString) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::Softmax, X, nullptr, -1);
    EXPECT_NE(op->toString().find("Softmax"), std::string::npos);
}

// ---- LogSoftmax tests ----

TEST_F(SoftmaxBasicTest, LogSoftmaxConstruction) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::LogSoftmax, X, nullptr, -1);

    EXPECT_EQ(op->getOpType(), OpType::LogSoftmax);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(SoftmaxBasicTest, LogSoftmaxShapeInference) {
    auto X = graph->addTensor({3, 5}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::LogSoftmax, X, nullptr, -1);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 2u);
    EXPECT_EQ(shapeVals[0], 3u);
    EXPECT_EQ(shapeVals[1], 5u);
}

TEST_F(SoftmaxBasicTest, LogSoftmaxToString) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<SoftmaxObj>(OpType::LogSoftmax, X, nullptr, -1);
    EXPECT_NE(op->toString().find("LogSoftmax"), std::string::npos);
}

} // namespace infini
