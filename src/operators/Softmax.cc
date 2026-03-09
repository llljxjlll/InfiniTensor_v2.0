#include "operators/Softmax.h"
#include "core/runtime.h"

namespace infini {

SoftmaxObj::SoftmaxObj(GraphObj *graph, OpType type, Tensor x, Tensor output,
                       int axis)
    : OperatorObj(type, TensorVec{x}, {output}), axis(axis) {
    IT_ASSERT(type == OpType::Softmax || type == OpType::LogSoftmax);
    IT_ASSERT(checkValid(graph));
}

string SoftmaxObj::toString() const {
    std::ostringstream os;
    os << type.toString() << "(x=" << inputs[0]->getGuid()
       << ",output=" << outputs[0]->getGuid() << ",axis=" << axis << ")";
    return os.str();
}

optional<vector<ShapeExpr>> SoftmaxObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> SoftmaxObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

SoftmaxObj::~SoftmaxObj() {
    if (infiniOpDesc) {
        infiniStatus_t err;
        if (type == OpType::Softmax) {
            err = infiniopDestroySoftmaxDescriptor(
                (infiniopSoftmaxDescriptor_t)infiniOpDesc);
        } else {
            err = infiniopDestroyLogSoftmaxDescriptor(
                (infiniopLogSoftmaxDescriptor_t)infiniOpDesc);
        }
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: " << type.toString()
                      << " descriptor destroy failed with error code " << err
                      << std::endl;
        }
    }
}

void SoftmaxObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xShape = inputs[0]->getShape();
    auto xStride = inputs[0]->getStride();

    infiniopTensorDescriptor_t yTensor, xTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));

    if (type == OpType::Softmax) {
        CHECK_INFINI_ERROR(infiniopCreateSoftmaxDescriptor(
            handle, (infiniopSoftmaxDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor, axis));
    } else {
        CHECK_INFINI_ERROR(infiniopCreateLogSoftmaxDescriptor(
            handle, (infiniopLogSoftmaxDescriptor_t *)&infiniOpDesc, yTensor,
            xTensor));
    }

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

} // namespace infini
