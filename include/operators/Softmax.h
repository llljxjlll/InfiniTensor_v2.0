#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/logsoftmax.h>
#include <infiniop/ops/softmax.h>

namespace infini {
/**
 * @brief Unified operator object for Softmax and LogSoftmax.
 *
 * The OpType passed at construction determines which InfiniCore kernel is used.
 * For LogSoftmax, the InfiniCore API has no axis parameter, so axis is only
 * used when opType == OpType::Softmax.
 */
class SoftmaxObj : public OperatorObj {
  private:
    int axis;

  public:
    /**
     * @param graph  The computation graph.
     * @param type   OpType::Softmax or OpType::LogSoftmax.
     * @param x      Input tensor.
     * @param output Output tensor, or nullptr to auto-create.
     * @param axis   Axis along which softmax is computed (default -1 = last).
     */
    SoftmaxObj(GraphObj *graph, OpType type, Tensor x, Tensor output,
               int axis = -1);

    string toString() const override;
    ~SoftmaxObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    int getAxis() const { return axis; }
};
} // namespace infini
