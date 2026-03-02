#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <infiniop/ops/lp_norm.h>

namespace infini {
class LpNormObj : public OperatorObj {
  private:
    int axis;
    int p;
    float eps;

  public:
    /**
     * @brief Construct a new LpNorm operator object.
     *
     * @param graph  The computation graph.
     * @param x      Input tensor.
     * @param output Output tensor, or nullptr to auto-create.
     * @param axis   Axis along which to compute the Lp norm.
     * @param p      The order of the norm (default 2 for L2).
     * @param eps    Small value for numerical stability.
     */
    LpNormObj(GraphObj *graph, Tensor x, Tensor output, int axis, int p = 2,
              float eps = 1e-12f);

    string toString() const override;
    ~LpNormObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    int getAxis() const { return axis; }
    int getP() const { return p; }
    float getEps() const { return eps; }
};
} // namespace infini
