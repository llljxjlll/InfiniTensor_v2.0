#include "operators/Conv.h"
#include "core/runtime.h"

namespace infini {

class ConvOpKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<ConvObj>(_op);
        op->createOpDesc();
        void *yData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        void *const wData = op->getInput(1)->getRawDataPtr<void *>();
        void *const bData = (op->getNumInputs() == 3)
                                ? op->getInput(2)->getRawDataPtr<void *>()
                                : nullptr;
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetConvWorkspaceSize(
            (infiniopConvDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(
            infiniopConv((infiniopConvDescriptor_t)op->getInfiniOpDesc(),
                         workspace, workspace_size, yData, xData, wData, bData,
                         runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Conv, ConvOpKernel);

} // namespace infini
