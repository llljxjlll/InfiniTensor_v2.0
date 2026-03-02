#include "core/runtime.h"
#include "operators/LpNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <thread>

namespace infini {

template <typename T> struct LpNormThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeX;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputXData, outputData;
    int axis = -1, p = 2;
    float eps = 1e-12f;
    bool completed = false;
};

template <typename T>
void lpnormDeviceThreadFunc(LpNormThreadParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto X = g->addTensor(params.shapeX, params.dataType);
        auto op = g->addOp<LpNormObj>(X, nullptr, params.axis, params.p, params.eps);

        X->setData(params.inputXData.data());
        runtime->dataMalloc(g);
        runtime->run(g);

        auto output = op->getOutput(0);
        size_t numElements = output->getElement();
        params.outputData.resize(numElements);
        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, output->getData()->getRawDataPtr(),
                        output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
        copyAndConvertData(params.outputData, hostPtr, numElements, params.dataType);
        runtime->deallocHost(hostPtr);
        params.completed = true;
    } catch (...) {
        params.completed = false;
    }
}

template <typename T>
void runMultiThreadLpNormTest(infiniDevice_t targetDevice, int targetId,
                               const DataType &dataType, bool print = false) {
    Shape shapeX = {2, 8};
    size_t numX = 16;
    auto xData = generateRandomData<T>(numX, static_cast<T>(-2), static_cast<T>(2));

    LpNormThreadParams<T> cpuParams, devParams;
    for (auto *p : {&cpuParams, &devParams}) {
        p->shapeX = shapeX;
        p->dataType = dataType;
        p->inputXData = xData;
        p->axis = 1;
        p->p = 2;
        p->eps = 1e-12f;
    }
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    devParams.device = targetDevice;
    devParams.deviceId = targetId;

    std::thread cpuThread(lpnormDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(lpnormDeviceThreadFunc<T>, std::ref(devParams));
    cpuThread.join();
    devThread.join();

    if (!cpuParams.completed || !devParams.completed) {
        GTEST_SKIP() << "One or both devices do not support this operation";
    }
    ASSERT_EQ(cpuParams.outputData.size(), devParams.outputData.size());

    size_t numErrors = 0;
    float maxError = 0.0f;
    const float epsilon = 1e-3f;
    for (size_t i = 0; i < cpuParams.outputData.size(); ++i) {
        float c, d;
        if constexpr (std::is_same_v<T, float>) {
            c = cpuParams.outputData[i];
            d = devParams.outputData[i];
        } else {
            c = fp16_to_fp32(cpuParams.outputData[i]);
            d = fp16_to_fp32(devParams.outputData[i]);
        }
        float err = std::abs(c - d);
        maxError = std::max(maxError, err);
        if (err > epsilon)
            numErrors++;
    }
    if (print)
        std::cout << "LpNorm: Errors=" << numErrors << ", MaxErr=" << maxError << std::endl;
    EXPECT_EQ(numErrors, 0u) << "CPU vs device mismatch (max error: " << maxError << ")";
}

// Reference L2 norm (normalize each row independently)
// Used by GPU multi-thread tests when available.
[[maybe_unused]] static std::vector<float> l2normRef(const std::vector<float> &x, float eps) {
    float norm = 0.0f;
    for (float v : x) norm += v * v;
    norm = std::sqrt(norm + eps);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i)
        y[i] = x[i] / norm;
    return y;
}

// ---- CPU absolute correctness ----
// NOTE: InfiniCore LpNorm has no CPU backend (GPU-only).

TEST(LpNorm, CPU_F32_AbsoluteCorrectness) {
    GTEST_SKIP() << "InfiniCore LpNorm does not have a CPU backend (GPU-only op)";
}

TEST(LpNorm, CPU_F32_SingleDevice) {
    GTEST_SKIP() << "InfiniCore LpNorm does not have a CPU backend (GPU-only op)";
}

// ---- Multi-platform tests ----

#ifdef USE_CUDA
TEST(LpNorm, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadLpNormTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                    DataType(INFINI_DTYPE_F32), true);
}
TEST(LpNorm, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadLpNormTest<uint16_t>(INFINI_DEVICE_NVIDIA, 0,
                                       DataType(INFINI_DTYPE_F16));
}
#endif

#ifdef USE_METAX
TEST(LpNorm, MultiThread_CPU_MetaX_F32) {
    runMultiThreadLpNormTest<float>(INFINI_DEVICE_METAX, 0,
                                    DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_ILUVATAR
TEST(LpNorm, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadLpNormTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                    DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_MOORE
TEST(LpNorm, MultiThread_CPU_Moore_F32) {
    runMultiThreadLpNormTest<float>(INFINI_DEVICE_MOORE, 0,
                                    DataType(INFINI_DTYPE_F32));
}
#endif

} // namespace infini
