#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/relu.h>
#include <infiniop/ops/sigmoid.h>
#include <infiniop/ops/silu.h>
#include <infiniop/ops/gelu.h>
#include <infiniop/ops/softplus.h>
#include <infiniop/ops/tanh.h>

namespace infini {
class UnaryObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Unary operator object.
     *
     * @param graph  The computation graph that this operator belongs to.
     * @param type   The operator type (Relu, Sigmoid, Silu, Gelu, Softplus, Tanh).
     * @param x      The input tensor.
     * @param output The output tensor (optional, pass nullptr to auto-create).
     */
    UnaryObj(GraphObj *graph, OpType type, Tensor x, Tensor output);
    string toString() const override;
    ~UnaryObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    OpType getUnaryOpType() const;
};
} // namespace infini
