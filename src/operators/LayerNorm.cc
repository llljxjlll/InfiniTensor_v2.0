#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

LayerNormObj::LayerNormObj(GraphObj *graph, Tensor x, Tensor weight,
                           Tensor bias, Tensor output, float eps)
    : OperatorObj(OpType::LayerNorm, TensorVec{x, weight, bias}, {output}),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LayerNormObj::toString() const {
    std::ostringstream os;
    os << "LayerNorm(x=" << inputs[0]->getGuid()
       << ",weight=" << inputs[1]->getGuid()
       << ",bias=" << inputs[2]->getGuid()
       << ",output=" << outputs[0]->getGuid() << ",eps=" << eps << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LayerNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LayerNormObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    IT_ASSERT(inputs[0]->getDataType() == inputs[2]->getDataType());
    return {inputs[0]->getDataType()};
}

LayerNormObj::~LayerNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyLayerNormDescriptor(
            (infiniopLayerNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr
                << "Warning: LayerNorm descriptor destroy failed with error "
                   "code "
                << err << std::endl;
        }
    }
}

void LayerNormObj::createOpDesc() {
    // InfiniCore CPU LayerNorm kernel internally indexes with 3 dimensions:
    //   [outer0, outer1, normalized_size]
    // Regardless of actual input rank, we pass canonical 3D descriptors.
    auto wShape = inputs[1]->getShape();
    auto wStride = inputs[1]->getStride();
    auto bShape = inputs[2]->getShape();
    auto bStride = inputs[2]->getStride();

    // normalized_size = product of last wNdim dims of x = weight element count
    size_t normalizedSize = inputs[1]->getElement();
    // othersize = product of remaining (batch) dims
    size_t otherSize = inputs[0]->getElement() / normalizedSize;
    infiniDtype_t dtype = inputs[0]->getDataType().getType();

    // Canonical 3D shape [1, otherSize, normalizedSize] with row-major strides
    size_t shape3D[3] = {1, otherSize, normalizedSize};
    ptrdiff_t stride3D[3] = {(ptrdiff_t)(otherSize * normalizedSize),
                              (ptrdiff_t)normalizedSize, 1};
    // Canonical 2D std_deviation shape [1, otherSize]
    size_t stdDevShape[2] = {1, otherSize};
    ptrdiff_t stdDevStride[2] = {(ptrdiff_t)otherSize, 1};

    infiniopTensorDescriptor_t outTensor, xTensor, wTensor, bTensor;
    infiniopTensorDescriptor_t stdXTensor, stdDevTensor;

    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &outTensor, 3, shape3D, stride3D, dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, 3, shape3D, stride3D, dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wTensor, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &bTensor, bShape->size(), bShape->getConstantValue().data(),
        bStride->getConstantValue().data(), dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &stdXTensor, 3, shape3D, stride3D, dtype));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &stdDevTensor, 2, stdDevShape, stdDevStride, dtype));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateLayerNormDescriptor(
        handle, (infiniopLayerNormDescriptor_t *)&infiniOpDesc, outTensor,
        stdXTensor, stdDevTensor, xTensor, wTensor, bTensor, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(outTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(stdXTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(stdDevTensor));
}

} // namespace infini
