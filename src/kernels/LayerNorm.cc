#include "operators/LayerNorm.h"
#include "core/runtime.h"

namespace infini {

class LayerNormOpKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<LayerNormObj>(_op);
        op->createOpDesc();
        void *outData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        void *const wData = op->getInput(1)->getRawDataPtr<void *>();
        void *const bData = op->getInput(2)->getRawDataPtr<void *>();

        size_t op_ws_size = 0;
        CHECK_INFINI_ERROR(infiniopGetLayerNormWorkspaceSize(
            (infiniopLayerNormDescriptor_t)op->getInfiniOpDesc(), &op_ws_size));

        // Compute intermediate buffer sizes for input_standardization and
        // input_std_deviation, both using the same dtype as x.
        size_t xTotalBytes = op->getInput(0)->getTotalBytes();
        size_t xElems = op->getInput(0)->getElement();
        size_t wElems = op->getInput(1)->getElement();
        size_t stdDevElems = (wElems > 0) ? (xElems / wElems) : 1;
        size_t elemBytes = (xElems > 0) ? (xTotalBytes / xElems) : sizeof(float);
        size_t std_x_bytes = xTotalBytes;             // same as x
        size_t std_dev_bytes = stdDevElems * elemBytes;

        void *workspace =
            runtime->getWorkspace(op_ws_size + std_x_bytes + std_dev_bytes);
        void *std_x_ptr = (char *)workspace + op_ws_size;
        void *std_dev_ptr = (char *)workspace + op_ws_size + std_x_bytes;

        CHECK_INFINI_ERROR(infiniopLayerNorm(
            (infiniopLayerNormDescriptor_t)op->getInfiniOpDesc(), workspace,
            op_ws_size, outData, std_x_ptr, std_dev_ptr, xData, wData, bData,
            runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::LayerNorm, LayerNormOpKernel);

} // namespace infini
