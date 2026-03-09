#include "core/runtime.h"
#include "operators/LpNorm.h"
#include "gtest/gtest.h"

namespace infini {

class LpNormBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(LpNormBasicTest, BasicConstruction) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 1, 2, 1e-12f);

    EXPECT_EQ(op->getOpType(), OpType::LpNorm);
    EXPECT_EQ(op->getNumInputs(), 1);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(LpNormBasicTest, ShapeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 1, 2, 1e-12f);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 2u);
    EXPECT_EQ(shapeVals[0], 2u);
    EXPECT_EQ(shapeVals[1], 4u);
}

TEST_F(LpNormBasicTest, ShapeInference3D) {
    auto X = graph->addTensor({1, 8, 16}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 2, 2, 1e-12f);

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 3u);
    EXPECT_EQ(shapeVals[0], 1u);
    EXPECT_EQ(shapeVals[1], 8u);
    EXPECT_EQ(shapeVals[2], 16u);
}

TEST_F(LpNormBasicTest, DataTypeInference) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 1, 2, 1e-12f);

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(LpNormBasicTest, ParamsStored) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 1, 2, 1e-12f);
    EXPECT_EQ(op->getAxis(), 1);
    EXPECT_EQ(op->getP(), 2);
}

TEST_F(LpNormBasicTest, ToString) {
    auto X = graph->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<LpNormObj>(X, nullptr, 1, 2, 1e-12f);
    EXPECT_NE(op->toString().find("LpNorm"), std::string::npos);
}

} // namespace infini
