#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/clip.h>

namespace infini {
class ClipObj : public OperatorObj {
  public:
    /**
     * @brief Construct a new Clip operator object.
     *
     * @param graph The computation graph that this operator belongs to.
     * @param x     The input tensor.
     * @param min_val The minimum value tensor (element-wise lower bound).
     * @param max_val The maximum value tensor (element-wise upper bound).
     * @param output  The output tensor (optional, pass nullptr to auto-create).
     */
    ClipObj(GraphObj *graph, Tensor x, Tensor min_val, Tensor max_val,
            Tensor output);
    string toString() const override;
    ~ClipObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;
};
} // namespace infini
