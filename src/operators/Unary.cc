#include "operators/Unary.h"
#include "core/runtime.h"

namespace infini {

UnaryObj::UnaryObj(GraphObj *graph, OpType type_, Tensor x, Tensor output)
    : OperatorObj(type_, TensorVec{x}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string UnaryObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "(";
    os << "x=" << inputs[0]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> UnaryObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> UnaryObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

UnaryObj::~UnaryObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = INFINI_STATUS_SUCCESS;
        if (type == OpType::Relu) {
            err = infiniopDestroyReluDescriptor(
                (infiniopReluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Sigmoid) {
            err = infiniopDestroySigmoidDescriptor(
                (infiniopSigmoidDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Silu) {
            err = infiniopDestroySiluDescriptor(
                (infiniopSiluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Gelu) {
            err = infiniopDestroyGeluDescriptor(
                (infiniopGeluDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Softplus) {
            err = infiniopDestroySoftplusDescriptor(
                (infiniopSoftplusDescriptor_t)infiniOpDesc);
        } else if (type == OpType::Tanh) {
            err = infiniopDestroyTanhDescriptor(
                (infiniopTanhDescriptor_t)infiniOpDesc);
        }
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: " << type.toString()
                      << " descriptor destroy failed with error code " << err
                      << std::endl;
        }
    }
}

void UnaryObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xStride = inputs[0]->getStride();

    auto ndim = yShape->size();
    auto yShapeVals = yShape->getConstantValue();
    auto yStrideVals = yStride->getConstantValue();
    auto xStrideVals = xStride->getConstantValue();

    infiniopTensorDescriptor_t yTensor, xTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, ndim, yShapeVals.data(), yStrideVals.data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, ndim, yShapeVals.data(), xStrideVals.data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    if (type == OpType::Relu) {
        CHECK_INFINI_ERROR(infiniopCreateReluDescriptor(
            handle, (infiniopReluDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else if (type == OpType::Sigmoid) {
        CHECK_INFINI_ERROR(infiniopCreateSigmoidDescriptor(
            handle, (infiniopSigmoidDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else if (type == OpType::Silu) {
        CHECK_INFINI_ERROR(infiniopCreateSiluDescriptor(
            handle, (infiniopSiluDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else if (type == OpType::Gelu) {
        CHECK_INFINI_ERROR(infiniopCreateGeluDescriptor(
            handle, (infiniopGeluDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else if (type == OpType::Softplus) {
        CHECK_INFINI_ERROR(infiniopCreateSoftplusDescriptor(
            handle, (infiniopSoftplusDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else if (type == OpType::Tanh) {
        CHECK_INFINI_ERROR(infiniopCreateTanhDescriptor(
            handle, (infiniopTanhDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    } else {
        IT_TODO_HALT_MSG("Unary operator not supported yet");
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

OpType UnaryObj::getUnaryOpType() const { return type; }

} // namespace infini
