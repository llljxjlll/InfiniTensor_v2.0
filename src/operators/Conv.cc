#include "operators/Conv.h"
#include "core/runtime.h"

namespace infini {

ConvObj::ConvObj(GraphObj *graph, Tensor x, Tensor w, Tensor b, Tensor output,
                 std::vector<int64_t> pads, std::vector<int64_t> strides,
                 std::vector<int64_t> dilations)
    : OperatorObj(OpType::Conv,
                  b ? TensorVec{x, w, b} : TensorVec{x, w}, {output}),
      pads(std::move(pads)), strides(std::move(strides)),
      dilations(std::move(dilations)) {
    IT_ASSERT(checkValid(graph));
}

string ConvObj::toString() const {
    std::ostringstream os;
    os << "Conv(x=" << inputs[0]->getGuid() << ",w=" << inputs[1]->getGuid();
    if (inputs.size() == 3)
        os << ",b=" << inputs[2]->getGuid();
    os << ",output=" << outputs[0]->getGuid() << ")";
    return os.str();
}

optional<vector<ShapeExpr>> ConvObj::inferShape() {
    auto xShape = inputs[0]->getShape();
    auto wShape = inputs[1]->getShape();
    IT_ASSERT(xShape->size() >= 3, "Conv input must have at least 3 dims");
    size_t n = xShape->size() - 2; // number of spatial dims
    IT_ASSERT(pads.size() == n && strides.size() == n &&
              dilations.size() == n);

    auto xVals = xShape->getConstantValue();
    auto wVals = wShape->getConstantValue();

    // Output: [N, C_out, out_1, ..., out_n]
    vector<Expr> outDims;
    outDims.push_back((*xShape)[0]);  // N
    outDims.push_back((*wShape)[0]);  // C_out
    for (size_t i = 0; i < n; ++i) {
        int64_t in_i = static_cast<int64_t>(xVals[2 + i]);
        int64_t k_i = static_cast<int64_t>(wVals[2 + i]);
        int64_t out_i = (in_i + 2 * pads[i] - dilations[i] * (k_i - 1) - 1) /
                            strides[i] +
                        1;
        outDims.push_back(ExprObj::constant(out_i));
    }
    return {{make_ref<ShapeExprObj>(outDims)}};
}

vector<DataType> ConvObj::inferDataType() const {
    IT_ASSERT(inputs[0]->getDataType() == inputs[1]->getDataType());
    return {inputs[0]->getDataType()};
}

ConvObj::~ConvObj() {
    if (infiniOpDesc) {
        infiniStatus_t err =
            infiniopDestroyConvDescriptor((infiniopConvDescriptor_t)infiniOpDesc);
        if (err != INFINI_STATUS_SUCCESS) {
            std::cerr << "Warning: Conv descriptor destroy failed with error "
                         "code "
                      << err << std::endl;
        }
    }
}

void ConvObj::createOpDesc() {
    auto yShape = outputs[0]->getShape();
    auto yStride = outputs[0]->getStride();
    auto xShape = inputs[0]->getShape();
    auto xStride = inputs[0]->getStride();
    auto wShape = inputs[1]->getShape();
    auto wStride = inputs[1]->getStride();

    infiniopTensorDescriptor_t yTensor, xTensor, wTensor, bTensor = nullptr;
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

    if (inputs.size() == 3 && inputs[2] != nullptr) {
        auto bShape = inputs[2]->getShape();
        auto bStride = inputs[2]->getStride();
        CHECK_INFINI_ERROR(infiniopCreateTensorDescriptor(
            &bTensor, bShape->size(), bShape->getConstantValue().data(),
            bStride->getConstantValue().data(),
            inputs[2]->getDataType().getType()));
    }

    infiniopHandle_t handle = nullptr;
    CHECK_INFINI_ERROR(infiniopCreateHandle(&handle));
    size_t n = pads.size();
    CHECK_INFINI_ERROR(infiniopCreateConvDescriptor(
        handle, (infiniopConvDescriptor_t *)&infiniOpDesc, yTensor, xTensor,
        wTensor, bTensor, (void *)pads.data(), (void *)strides.data(),
        (void *)dilations.data(), n));

    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(yTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(xTensor));
    CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(wTensor));
    if (bTensor)
        CHECK_INFINI_ERROR(infiniopDestroyTensorDescriptor(bTensor));
}

} // namespace infini
