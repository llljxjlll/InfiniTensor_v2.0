#include "operators/Softmax.h"
#include "core/runtime.h"

namespace infini {

class SoftmaxOpKernel : public Kernel {
    void compute(const Operator &_op,
                 const RuntimeObj *runtime) const override {
        auto op = as<SoftmaxObj>(_op);
        op->createOpDesc();
        void *yData = op->getOutput(0)->getRawDataPtr<void *>();
        void *const xData = op->getInput(0)->getRawDataPtr<void *>();
        void *stream = runtime->getCurrentThreadContext()->stream;

        if (op->getOpType() == OpType::Softmax) {
            size_t workspace_size = 0;
            CHECK_INFINI_ERROR(infiniopGetSoftmaxWorkspaceSize(
                (infiniopSoftmaxDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(
                infiniopSoftmax((infiniopSoftmaxDescriptor_t)op->getInfiniOpDesc(),
                                workspace, workspace_size, yData, xData, stream));
        } else {
            size_t workspace_size = 0;
            CHECK_INFINI_ERROR(infiniopGetLogSoftmaxWorkspaceSize(
                (infiniopLogSoftmaxDescriptor_t)op->getInfiniOpDesc(),
                &workspace_size));
            void *workspace = runtime->getWorkspace(workspace_size);
            CHECK_INFINI_ERROR(infiniopLogSoftmax(
                (infiniopLogSoftmaxDescriptor_t)op->getInfiniOpDesc(), workspace,
                workspace_size, yData, xData, stream));
        }
    }
};

REGISTER_KERNEL_ALL_DEVICES(OpType::Softmax, SoftmaxOpKernel);
REGISTER_KERNEL_ALL_DEVICES(OpType::LogSoftmax, SoftmaxOpKernel);

} // namespace infini
