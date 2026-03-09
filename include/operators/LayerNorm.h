#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/layer_norm.h>

namespace infini {
class LayerNormObj : public OperatorObj {
  private:
    float eps;

  public:
    /**
     * @brief Construct a new LayerNorm operator object.
     *
     * @param graph   The computation graph.
     * @param x       Input tensor.
     * @param weight  Scale tensor (same shape as the last normalized dims).
     * @param bias    Bias tensor (same shape as weight).
     * @param output  Output tensor, or nullptr to auto-create.
     * @param eps     Small value for numerical stability.
     */
    LayerNormObj(GraphObj *graph, Tensor x, Tensor weight, Tensor bias,
                 Tensor output, float eps = 1e-5f);

    string toString() const override;
    ~LayerNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    float getEps() const { return eps; }
};
} // namespace infini
