#include "core/runtime.h"
#include "operators/Conv.h"
#include "gtest/gtest.h"

namespace infini {

class ConvBasicTest : public testing::Test {
  protected:
    Runtime runtime;
    Graph graph;

    void SetUp() override {
        runtime = make_ref<RuntimeObj>();
        graph = make_ref<GraphObj>(runtime);
    }
};

TEST_F(ConvBasicTest, BasicConstruction) {
    // x=[1,1,5,5], w=[1,1,3,3], no bias
    auto X = graph->addTensor({1, 1, 5, 5}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({1, 1, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, nullptr, nullptr,
                                    std::vector<int64_t>{0, 0},
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{1, 1});
    EXPECT_EQ(op->getOpType(), OpType::Conv);
    EXPECT_EQ(op->getNumInputs(), 2);
    EXPECT_EQ(op->getNumOutputs(), 1);
}

TEST_F(ConvBasicTest, BasicConstructionWithBias) {
    auto X = graph->addTensor({1, 1, 5, 5}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({1, 1, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto B = graph->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, B, nullptr,
                                    std::vector<int64_t>{0, 0},
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{1, 1});
    EXPECT_EQ(op->getNumInputs(), 3);
}

TEST_F(ConvBasicTest, ShapeInference) {
    // x=[1,1,5,5], w=[1,1,3,3], pads=[0,0], strides=[1,1], dilations=[1,1]
    // out_h = (5 + 0 - 1*(3-1) - 1) / 1 + 1 = 3
    auto X = graph->addTensor({1, 1, 5, 5}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({1, 1, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, nullptr, nullptr,
                                    std::vector<int64_t>{0, 0},
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{1, 1});

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 4u);
    EXPECT_EQ(shapeVals[0], 1u);  // N
    EXPECT_EQ(shapeVals[1], 1u);  // C_out
    EXPECT_EQ(shapeVals[2], 3u);  // H_out
    EXPECT_EQ(shapeVals[3], 3u);  // W_out
}

TEST_F(ConvBasicTest, ShapeInferenceWithPaddingAndStride) {
    // x=[1,3,7,7], w=[8,3,3,3], pads=[1,1], strides=[2,2], dilations=[1,1]
    // out_h = (7 + 2 - 1*(3-1) - 1) / 2 + 1 = (7+2-2-1)/2+1 = 6/2+1 = 4
    auto X = graph->addTensor({1, 3, 7, 7}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({8, 3, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, nullptr, nullptr,
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{2, 2},
                                    std::vector<int64_t>{1, 1});

    auto inferredShapes = op->inferShape();
    ASSERT_TRUE(inferredShapes.has_value());
    auto shapeVals = (*inferredShapes)[0]->getConstantValue();
    ASSERT_EQ(shapeVals.size(), 4u);
    EXPECT_EQ(shapeVals[0], 1u);
    EXPECT_EQ(shapeVals[1], 8u);
    EXPECT_EQ(shapeVals[2], 4u);
    EXPECT_EQ(shapeVals[3], 4u);
}

TEST_F(ConvBasicTest, DataTypeInference) {
    auto X = graph->addTensor({1, 1, 4, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({1, 1, 2, 2}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, nullptr, nullptr,
                                    std::vector<int64_t>{0, 0},
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{1, 1});

    auto inferredTypes = op->inferDataType();
    ASSERT_EQ(inferredTypes.size(), 1u);
    EXPECT_EQ(inferredTypes[0], DataType(INFINI_DTYPE_F32));
}

TEST_F(ConvBasicTest, ToString) {
    auto X = graph->addTensor({1, 1, 4, 4}, DataType(INFINI_DTYPE_F32));
    auto W = graph->addTensor({1, 1, 2, 2}, DataType(INFINI_DTYPE_F32));
    auto op = graph->addOp<ConvObj>(X, W, nullptr, nullptr,
                                    std::vector<int64_t>{0, 0},
                                    std::vector<int64_t>{1, 1},
                                    std::vector<int64_t>{1, 1});
    EXPECT_NE(op->toString().find("Conv"), std::string::npos);
}

} // namespace infini
