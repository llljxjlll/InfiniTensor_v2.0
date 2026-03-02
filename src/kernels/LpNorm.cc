#include "operators/LpNorm.h"
#include "core/runtime.h"

namespace infini {

class LpNormOpKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<LpNormObj>(_op);
        op->createOpDesc();
        void *outData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        size_t workspace_size = 0;
        CHECK_INFINI_ERROR(infiniopGetLPNormWorkspaceSize(
            (infiniopLPNormDescriptor_t)op->getInfiniOpDesc(), &workspace_size));
        void *workspace = runtime->getWorkspace(workspace_size);
        CHECK_INFINI_ERROR(infiniopLPNorm(
            (infiniopLPNormDescriptor_t)op->getInfiniOpDesc(), workspace,
            workspace_size, outData, xData,
            runtime->getCurrentThreadContext()->stream));
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::LpNorm, LpNormOpKernel);

} // namespace infini
