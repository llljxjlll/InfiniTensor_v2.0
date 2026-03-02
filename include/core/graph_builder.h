#pragma once
#ifndef GRAPH_BUILDER_H
#define GRAPH_BUILDER_H

#include "core/graph.h"
#include "core/op_type.h"
#include "operators/Clip.h"
#include "operators/Conv.h"
#include "operators/ElementWise.h"
#include "operators/Gemm.h"
#include "operators/LayerNorm.h"
#include "operators/LpNorm.h"
#include "operators/RmsNorm.h"
#include "operators/Softmax.h"
#include "operators/Unary.h"

namespace infini {

class GraphBuilderObj {
  private:
    Ref<GraphObj> g;

  public:
    GraphBuilderObj(Runtime runtime);

    Tensor tensor(ShapeExpr dims, DataType dtype,
                  std::optional<StrideExpr> stride = std::nullopt);

    Tensor gemm(Tensor A, Tensor B, Tensor C, float alpha = 1.0,
                float beta = 1.0, bool transA = false, bool transB = false,
                std::optional<Tensor> Y = std::nullopt);
    Tensor clip(Tensor X, Tensor min_val, Tensor max_val,
                std::optional<Tensor> Y = std::nullopt);
    Tensor add(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor sub(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);
    Tensor mul(Tensor A, Tensor B, std::optional<Tensor> Y = std::nullopt);

    Tensor conv(Tensor x, Tensor w, Tensor b, std::vector<int64_t> pads,
                std::vector<int64_t> strides, std::vector<int64_t> dilations,
                std::optional<Tensor> Y = std::nullopt);
    Tensor layernorm(Tensor x, Tensor weight, Tensor bias, float eps = 1e-5f,
                     std::optional<Tensor> Y = std::nullopt);
    Tensor lpnorm(Tensor x, int axis, int p = 2, float eps = 1e-12f,
                  std::optional<Tensor> Y = std::nullopt);
    Tensor rmsnorm(Tensor x, Tensor w, float epsilon = 1e-6f,
                   std::optional<Tensor> Y = std::nullopt);
    Tensor softmax(Tensor x, int axis = -1,
                   std::optional<Tensor> Y = std::nullopt);
    Tensor logsoftmax(Tensor x, std::optional<Tensor> Y = std::nullopt);

    Tensor relu(Tensor x, std::optional<Tensor> Y = std::nullopt);
    Tensor sigmoid(Tensor x, std::optional<Tensor> Y = std::nullopt);
    Tensor silu(Tensor x, std::optional<Tensor> Y = std::nullopt);
    Tensor gelu(Tensor x, std::optional<Tensor> Y = std::nullopt);
    Tensor softplus(Tensor x, std::optional<Tensor> Y = std::nullopt);
    Tensor tanh(Tensor x, std::optional<Tensor> Y = std::nullopt);

    string printGraph() const;

    Graph getGraph() const;
};

} // namespace infini
#endif // GRAPH_BUILDER_H
