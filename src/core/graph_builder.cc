#include "core/graph_builder.h"

namespace infini {

GraphBuilderObj::GraphBuilderObj(Runtime runtime)
    : g(make_ref<GraphObj>(std::move(runtime))) {}

Tensor GraphBuilderObj::tensor(ShapeExpr dims, DataType dtype,
                               std::optional<StrideExpr> stride) {
    if (stride.has_value()) {
        return g->addTensor(dims, stride.value(), dtype);
    } else {
        return g->addTensor(dims, dtype);
    }
}

Tensor GraphBuilderObj::gemm(Tensor A, Tensor B, Tensor C, float alpha,
                             float beta, bool transA, bool transB,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        g->addOpWithOutputs<GemmObj>(std::move(A), std::move(B),
                                     std::move(Y.value()), std::move(C), alpha,
                                     beta, transA, transB);
        return Y.value();
    } else {
        return g
            ->addOp<GemmObj>(std::move(A), std::move(B), nullptr, std::move(C),
                             alpha, beta, transA, transB)
            ->getOutput(0);
    }
}

#define DEFINE_BINARY_OP(OP, TYPE)                                             \
    Tensor GraphBuilderObj::OP(Tensor A, Tensor B, std::optional<Tensor> Y) {  \
        if (Y.has_value()) {                                                   \
            g->addOpWithOutputs<ElementWiseObj>(                               \
                TYPE, std::move(A), std::move(B), std::move(Y.value()));       \
            return Y.value();                                                  \
        } else {                                                               \
            return g                                                           \
                ->addOp<ElementWiseObj>(TYPE, std::move(A), std::move(B),      \
                                        nullptr)                               \
                ->getOutput(0);                                                \
        }                                                                      \
    }

DEFINE_BINARY_OP(add, OpType::Add);
DEFINE_BINARY_OP(sub, OpType::Sub);
DEFINE_BINARY_OP(mul, OpType::Mul);

Tensor GraphBuilderObj::clip(Tensor X, Tensor min_val, Tensor max_val,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value(); // save before move
        g->addOpWithOutputs<ClipObj>(std::move(X), std::move(min_val),
                                     std::move(max_val), y_out);
        return y_out;
    } else {
        return g->addOp<ClipObj>(std::move(X), std::move(min_val),
                                  std::move(max_val), nullptr)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::conv(Tensor x, Tensor w, Tensor b,
                             std::vector<int64_t> pads,
                             std::vector<int64_t> strides,
                             std::vector<int64_t> dilations,
                             std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<ConvObj>(std::move(x), std::move(w), std::move(b),
                                     y_out, std::move(pads), std::move(strides),
                                     std::move(dilations));
        return y_out;
    } else {
        return g->addOp<ConvObj>(std::move(x), std::move(w), std::move(b),
                                  nullptr, std::move(pads), std::move(strides),
                                  std::move(dilations))
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::layernorm(Tensor x, Tensor weight, Tensor bias,
                                  float eps, std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<LayerNormObj>(std::move(x), std::move(weight),
                                          std::move(bias), y_out, eps);
        return y_out;
    } else {
        return g->addOp<LayerNormObj>(std::move(x), std::move(weight),
                                      std::move(bias), nullptr, eps)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::lpnorm(Tensor x, int axis, int p, float eps,
                               std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<LpNormObj>(std::move(x), y_out, axis, p, eps);
        return y_out;
    } else {
        return g->addOp<LpNormObj>(std::move(x), nullptr, axis, p, eps)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::rmsnorm(Tensor x, Tensor w, float epsilon,
                                std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<RmsNormObj>(std::move(x), std::move(w), y_out,
                                        epsilon);
        return y_out;
    } else {
        return g->addOp<RmsNormObj>(std::move(x), std::move(w), nullptr,
                                     epsilon)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::softmax(Tensor x, int axis, std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<SoftmaxObj>(OpType::Softmax, std::move(x), y_out,
                                        axis);
        return y_out;
    } else {
        return g->addOp<SoftmaxObj>(OpType::Softmax, std::move(x), nullptr,
                                     axis)
            ->getOutput(0);
    }
}

Tensor GraphBuilderObj::logsoftmax(Tensor x, std::optional<Tensor> Y) {
    if (Y.has_value()) {
        Tensor y_out = Y.value();
        g->addOpWithOutputs<SoftmaxObj>(OpType::LogSoftmax, std::move(x),
                                        y_out, -1);
        return y_out;
    } else {
        return g->addOp<SoftmaxObj>(OpType::LogSoftmax, std::move(x), nullptr,
                                     -1)
            ->getOutput(0);
    }
}

string GraphBuilderObj::printGraph() const { return g->toString(); }

Graph GraphBuilderObj::getGraph() const { return g; }
} // namespace infini
