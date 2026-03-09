#pragma once
#include "core/graph.h"
#include "core/operator.h"
#include <cstdint>
#include <infiniop/ops/conv.h>

namespace infini {
class ConvObj : public OperatorObj {
  private:
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;

  public:
    /**
     * @brief Construct a new Conv operator object.
     *
     * @param graph    The computation graph.
     * @param x        Input tensor [N, C_in, d1, ..., dn].
     * @param w        Weight tensor [C_out, C_in, k1, ..., kn].
     * @param b        Bias tensor [C_out], or nullptr for no bias.
     * @param output   Output tensor, or nullptr to auto-create.
     * @param pads     Padding for each spatial dim (size n).
     * @param strides  Stride for each spatial dim (size n).
     * @param dilations Dilation for each spatial dim (size n).
     */
    ConvObj(GraphObj *graph, Tensor x, Tensor w, Tensor b, Tensor output,
            std::vector<int64_t> pads, std::vector<int64_t> strides,
            std::vector<int64_t> dilations);

    string toString() const override;
    ~ConvObj() override;

    void createOpDesc() override;
    optional<vector<ShapeExpr>> inferShape() override;
    vector<DataType> inferDataType() const override;

    const std::vector<int64_t> &getPads() const { return pads; }
    const std::vector<int64_t> &getStrides() const { return strides; }
    const std::vector<int64_t> &getDilations() const { return dilations; }
};
} // namespace infini
