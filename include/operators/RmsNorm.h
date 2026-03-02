#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/rms_norm.h>

namespace infini {
class RmsNormObj : public OperatorObj {
  private:
    float epsilon;

  public:
    /**
     * @brief Construct a new RmsNorm operator object.
     *
     * @param graph   The computation graph.
     * @param x       Input tensor.
     * @param w       Weight tensor (same shape as the last dim of x).
     * @param output  Output tensor, or nullptr to auto-create.
     * @param epsilon Small value for numerical stability.
     */
    RmsNormObj(GraphObj *graph, Tensor x, Tensor w, Tensor output,
               float epsilon = 1e-6f);

    string toString() const override;
    ~RmsNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    float getEpsilon() const { return epsilon; }
};
} // namespace infini
