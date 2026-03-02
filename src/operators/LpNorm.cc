#include "operators/LpNorm.h"
#include "core/runtime.h"

namespace infini {

LpNormObj::LpNormObj(GraphObj *graph, Tensor x, Tensor output, int axis, int p,
                     float eps)
    : OperatorObj(OpType::LpNorm, TensorVec{x}, {output}), axis(axis), p(p),
      eps(eps) {
    IT_ASSERT(checkValid(graph));
}

string LpNormObj::toString() const {
    std::ostringstream os;
    os << "LpNorm(x=" << inputs[0]->getGuid()
       << ",output=" << outputs[0]->getGuid() << ",axis=" << axis << ",p=" << p
       << ",eps=" << eps << ")";
    return os.str();
}

optional<vector<ShapeExpr>> LpNormObj::inferShape() {
    return {{inputs[0]->getShape()}};
}

vector<DataType> LpNormObj::inferDataType() const {
    return {inputs[0]->getDataType()};
}

LpNormObj::~LpNormObj() {
    if (infiniOpDesc) {
        infiniStatus_t err = infiniopDestroyLPNormDescriptor(
            (infiniopLPNormDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr
                << "Warning: LpNorm descriptor destroy failed with error code "
                << err << std::endl;
        }
    }
}

void LpNormObj::createOpDesc() {
    auto outShape = outputs[0]->getShape();
    auto outStride = outputs[0]->getStride();
    auto xShape = inputs[0]->getShape();
    auto xStride = inputs[0]->getStride();

    infiniopTensorDescriptor_t outTensor, xTensor;
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &outTensor, outShape->size(), outShape->getConstantValue().data(),
        outStride->getConstantValue().data(),
        outputs[0]->getDataType().getType()));
    CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
        &xTensor, xShape->size(), xShape->getConstantValue().data(),
        xStride->getConstantValue().data(),
        inputs[0]->getDataType().getType()));

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    CHECK_INFINI_ERROR(infiniopCreateLPNormDescriptor(
        handle, (infiniopLPNormDescriptor_t *)&infiniOpDesc, outTensor, xTensor,
        axis, p, eps));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(outTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
}

} // namespace infini
