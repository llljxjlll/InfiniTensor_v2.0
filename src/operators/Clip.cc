#include "operators/Clip.h"
#include "core/runtime.h"

namespace infini {

ClipObj::ClipObj(GraphObj *graph, Tensor x, Tensor min_val, Tensor max_val,
                 Tensor output)
    : OperatorObj(OpType::Clip, TensorVec{x, min_val, max_val}, {output}) {
    IT_ASSERT(checkValid(graph));
}

string ClipObj::toString() const {
    std::ostringstream os;
    os << "Clip(";
    os << "x=" << inputs[0]->getGuid() << ",";
    os << "min_val=" << inputs[1]->getGuid() << ",";
    os << "max_val=" << inputs[2]->getGuid() << ",";
    os << "output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> ClipObj::inferShape() {
    // Output shape equals the input x shape (clip is element-wise on x).
    return {{inputs[0]->getShape()}};
}

vector<DataType> ClipObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    IT_ASSERT(inputs[0]->getDataType() == inputs[2]->getDataType());
    return {inputs[0]->getDataType()};
}

ClipObj::~ClipObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyClipDescriptor(
            (infiniopClipDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr
                << "Warning: Clip descriptor destroy failed with error code "
                << err << std::endl;
        }
    }
}

void ClipObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();

    // Compute broadcast strides for x, min_val, max_val relative to y's shape.
    auto xStride =
        broadcastStride(inputs[0]->getShape(), inputs[0]->getStride(), yShape);
    auto minStride =
        broadcastStride(inputs[1]->getShape(), inputs[1]->getStride(), yShape);
    auto maxStride =
        broadcastStride(inputs[2]->getShape(), inputs[2]->getStride(), yShape);

    auto ndim = yShape->size();
    auto yShapeVals = yShape->getConstantValue();
    auto yStrideVals = yStride->getConstantValue();
    auto xStrideVals = xStride->getConstantValue();
    auto minStrideVals = minStride->getConstantValue();
    auto maxStrideVals = maxStride->getConstantValue();

    infiniopTensorDescriptor_t yTensor, xTensor, minTensor, maxTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, ndim, yShapeVals.data(), yStrideVals.data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, ndim, yShapeVals.data(), xStrideVals.data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &minTensor, ndim, yShapeVals.data(), minStrideVals.data(),
        inputs[1]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &maxTensor, ndim, yShapeVals.data(), maxStrideVals.data(),
        inputs[2]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateClipDescriptor(
        handle, (infiniopClipDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        minTensor, maxTensor));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(minTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(maxTensor));
}

} // namespace infini
