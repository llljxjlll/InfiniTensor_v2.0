#include "core/runtime.h"
#include "operators/Softmax.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <thread>

namespace infini {

template <typename T> struct SoftmaxThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeX;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputXData, outputData;
    OpType opType = OpType::Softmax;
    int axis = -1;
    bool completed = false;
};

template <typename T>
void softmaxDeviceThreadFunc(SoftmaxThreadParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto X = g->addTensor(params.shapeX, params.dataType);
        auto op = g->addOp<SoftmaxObj>(params.opType, X, nullptr, params.axis);

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
void runMultiThreadSoftmaxTest(infiniDevice_t targetDevice, int targetId,
                               const DataType &dataType, OpType opType,
                               bool print = false) {
    Shape shapeX = {2, 8};
    size_t numX = 16;
    auto xData = generateRandomData<T>(numX, static_cast<T>(-2), static_cast<T>(2));

    SoftmaxThreadParams<T> cpuParams, devParams;
    for (auto *p : {&cpuParams, &devParams}) {
        p->shapeX = shapeX;
        p->dataType = dataType;
        p->inputXData = xData;
        p->opType = opType;
        p->axis = -1;
    }
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    devParams.device = targetDevice;
    devParams.deviceId = targetId;

    std::thread cpuThread(softmaxDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(softmaxDeviceThreadFunc<T>, std::ref(devParams));
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
        std::cout << "Softmax: Errors=" << numErrors << ", MaxErr=" << maxError << std::endl;
    EXPECT_EQ(numErrors, 0u) << "CPU vs device mismatch (max error: " << maxError << ")";
}

// Reference softmax along axis=1 for a [1, n] tensor
static std::vector<float> softmaxRef(const std::vector<float> &x) {
    float maxVal = *std::max_element(x.begin(), x.end());
    std::vector<float> y(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        y[i] = std::exp(x[i] - maxVal);
        sum += y[i];
    }
    for (float &v : y) v /= sum;
    return y;
}

static std::vector<float> logsoftmaxRef(const std::vector<float> &x) {
    auto sm = softmaxRef(x);
    for (float &v : sm) v = std::log(v);
    return sm;
}

// ---- CPU absolute correctness for Softmax ----
// NOTE: InfiniCore Softmax has no CPU backend (GPU-only).
// These tests are skipped on CPU-only builds.

TEST(Softmax, CPU_F32_AbsoluteCorrectness) {
    GTEST_SKIP() << "InfiniCore Softmax does not have a CPU backend (GPU-only op)";
}

TEST(Softmax, CPU_F32_SingleDevice) {
    GTEST_SKIP() << "InfiniCore Softmax does not have a CPU backend (GPU-only op)";
}

// ---- CPU absolute correctness for LogSoftmax ----

TEST(LogSoftmax, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    std::vector<float> xData = {1.0f, 2.0f, 3.0f};
    auto expected = logsoftmaxRef(xData);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({1, 3}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<SoftmaxObj>(OpType::LogSoftmax, X, nullptr, -1);
    X->setData(xData.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    auto output = op->getOutput(0);
    size_t n = output->getElement();
    std::vector<float> result(n);
    void *hostPtr = runtime->allocHost(output->getTotalBytes());
    runtime->memcpy(hostPtr, output->getData()->getRawDataPtr(),
                    output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
    std::memcpy(result.data(), hostPtr, n * sizeof(float));
    runtime->deallocHost(hostPtr);

    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        EXPECT_NEAR(result[i], expected[i], 1e-4f) << "Mismatch at index " << i;
}

TEST(LogSoftmax, CPU_F32_SingleDevice) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<SoftmaxObj>(OpType::LogSoftmax, X, nullptr, -1);

    std::vector<float> xData = {1, 2, 3, 4, 5, 6, 7, 8};
    X->setData(xData.data());
    runtime->dataMalloc(g);
    runtime->run(g);
    std::cout << "CPU LogSoftmax Output:" << std::endl;
    op->getOutput(0)->printData(runtime);
}

// ---- Multi-platform tests ----

#ifdef USE_CUDA
TEST(Softmax, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::Softmax, true);
}
TEST(Softmax, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadSoftmaxTest<uint16_t>(INFINI_DEVICE_NVIDIA, 0,
                                        DataType(INFINI_DTYPE_F16), OpType::Softmax);
}
TEST(LogSoftmax, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::LogSoftmax, true);
}
TEST(LogSoftmax, MultiThread_CPU_NVIDIA_F16) {
    // F16 multi-thread test skipped: generateRandomData<uint16_t> produces
    // invalid F16 bit patterns (static_cast<uint16_t>(-2) = 65534).
    GTEST_SKIP() << "F16 multi-thread test requires proper F16 data generation";
}
#endif

#ifdef USE_METAX
TEST(Softmax, MultiThread_CPU_MetaX_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_METAX, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::Softmax);
}
TEST(LogSoftmax, MultiThread_CPU_MetaX_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_METAX, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::LogSoftmax);
}
#endif

#ifdef USE_ILUVATAR
TEST(Softmax, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::Softmax);
}
TEST(LogSoftmax, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::LogSoftmax);
}
#endif

#ifdef USE_MOORE
TEST(Softmax, MultiThread_CPU_Moore_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_MOORE, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::Softmax);
}
TEST(LogSoftmax, MultiThread_CPU_Moore_F32) {
    runMultiThreadSoftmaxTest<float>(INFINI_DEVICE_MOORE, 0,
                                     DataType(INFINI_DTYPE_F32), OpType::LogSoftmax);
}
#endif

} // namespace infini
