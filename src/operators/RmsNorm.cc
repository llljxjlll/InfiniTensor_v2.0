#include "operators/RmsNorm.h"
#include "core/runtime.h"

namespace infini {

RmsNormObj::RmsNormObj(GraphObj *graph, Tensor x, Tensor w, Tensor output,
                       float epsilon)
    : OperatorObj(OpType::RmsNorm, TensorVec{x, w}, {output}),
      epsilon(epsilon) {
    IT_ASSERT(checkValid(graph));
}

string RmsNormObj::toString() const {
    std::ostringstream os;
    os << "RmsNorm(x=" << inputs[0]->getGuid()
       << ",w=" << inputs[1]->getGuid()
       << ",output=" << outputs[0]->getGuid() << ",epsilon=" << epsilon << ")";
    return os.str();
}

optional<vector<ShapeExpr>> RmsNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> RmsNormObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    return {inputs[0]->getDataType()};
}

RmsNormObj::~RmsNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyRMSNormDescriptor(
            (infiniopRMSNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr
                << "Warning: RmsNorm descriptor destroy failed with error code "
                << err << std::endl;
        }
    }
}

void RmsNormObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xShape = inputs[0]->getShape();
    auto xStride = inputs[0]->getStride();
    auto wShape = inputs[1]->getShape();
    auto wStride = inputs[1]->getStride();

    infiniopTensorDescriptor_t yTensor, xTensor, wTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &yTensor, yShape->size(), yShape->getConstantValue().data(),
        yStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &wTensor, wShape->size(), wShape->getConstantValue().data(),
        wStride->getConstantValue().data(),
        inputs[1]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateRMSNormDescriptor(
        handle, (infiniopRMSNormDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        wTensor, epsilon));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
}

} // namespace infini
