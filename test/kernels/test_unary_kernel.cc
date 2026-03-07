#include "core/runtime.h"
#include "operators/Unary.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <thread>

namespace infini {

// Per-device thread parameters for Unary ops
template <typename T> struct UnaryThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    OpType opType = OpType::Relu;
    Shape shape;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Execute a Unary op on a specific device and collect output
template <typename T>
void unaryDeviceThreadFunc(UnaryThreadParams<T> &params) {
    try {
        RuntimeObj::init();
        Runtime &runtime = RuntimeObj::getInstance();
        runtime->initThreadContext(params.device, params.deviceId);

        Graph g = make_ref<GraphObj>(runtime);
        auto X = g->addTensor(params.shape, params.dataType);
        auto op = g->addOp<UnaryObj>(params.opType, X, nullptr);

        X->setData(params.inputData.data());
        runtime->dataMalloc(g);
        runtime->run(g);

        auto output = op->getOutput(0);
        size_t numElements = output->getElement();
        params.outputData.resize(numElements);

        void *hostPtr = runtime->allocHost(output->getTotalBytes());
        runtime->memcpy(hostPtr, output->getData()->getRawDataPtr(),
                        output->getTotalBytes(), INFINIRT_MEMCPY_D2H);
        copyAndConvertData(params.outputData, hostPtr, numElements,
                           params.dataType);
        runtime->deallocHost(hostPtr);
        params.completed = true;
    } catch (...) {
        params.completed = false;
    }
}

// Run a Unary op on a single device, return float result vector
std::vector<float> runUnaryOnDevice(infiniDevice_t device, int deviceId,
                                    OpType opType,
                                    const std::vector<float> &xData,
                                    const Shape &shape) {
    UnaryThreadParams<float> params;
    params.device = device;
    params.deviceId = deviceId;
    params.opType = opType;
    params.shape = shape;
    params.dataType = DataType(INFINI_DTYPE_F32);
    params.inputData = xData;

    unaryDeviceThreadFunc(params);
    return params.outputData;
}

// Run Unary op on two devices in parallel and compare results
template <typename T>
void runMultiThreadUnaryTest(OpType opType, infiniDevice_t targetDevice,
                             int targetId, const DataType &dataType,
                             bool print = false) {
    Shape shape = {2, 3, 4};
    size_t numElems = 1;
    for (auto d : shape)
        numElems *= d;

    // Use small values in [-2, 2] for numerically stable activations
    std::vector<T> xData;
    if constexpr (std::is_same_v<T, uint16_t>) {
        auto xF = generateRandomData<float>(numElems, -2.0f, 2.0f);
        xData.resize(numElems);
        for (size_t i = 0; i < numElems; ++i) xData[i] = fp32_to_fp16(xF[i]);
    } else {
        xData = generateRandomData<T>(numElems, static_cast<T>(-2), static_cast<T>(2));
    }

    UnaryThreadParams<T> cpuParams, devParams;

    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.opType = opType;
    cpuParams.shape = shape;
    cpuParams.dataType = dataType;
    cpuParams.inputData = xData;
    cpuParams.deviceName = "CPU";

    devParams.device = targetDevice;
    devParams.deviceId = targetId;
    devParams.opType = opType;
    devParams.shape = shape;
    devParams.dataType = dataType;
    devParams.inputData = xData;

    if (print) {
        std::cout << "=======================================" << std::endl;
        std::cout << "Multi-Thread Unary Test: "
                  << OpType(opType).toString() << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "=======================================" << std::endl;
    }

    std::thread cpuThread(unaryDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(unaryDeviceThreadFunc<T>, std::ref(devParams));
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
        float cpuVal, devVal;
        if constexpr (std::is_same_v<T, float>) {
            cpuVal = cpuParams.outputData[i];
            devVal = devParams.outputData[i];
        } else if constexpr (std::is_same_v<T, uint16_t>) {
            cpuVal = fp16_to_fp32(cpuParams.outputData[i]);
            devVal = fp16_to_fp32(devParams.outputData[i]);
        }
        float err = std::abs(cpuVal - devVal);
        maxError = std::max(maxError, err);
        if (err > epsilon) {
            numErrors++;
            if (numErrors <= 5) {
                std::cout << "Mismatch at index " << i << ": CPU=" << cpuVal
                          << ", Device=" << devVal << ", error=" << err
                          << std::endl;
            }
        }
    }

    if (print) {
        std::cout << "Errors: " << numErrors << ", Max error: " << maxError
                  << std::endl;
        std::cout << (numErrors == 0 ? "✓ PASSED" : "✗ FAILED") << std::endl;
        std::cout << "=======================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0u)
        << "Results mismatch between CPU and device (max error: " << maxError
        << ")";
}

// ---- CPU absolute correctness tests ----

TEST(Relu, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // relu(x) = max(x, 0)
    std::vector<float> xData = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 1.0f, 2.0f};

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Relu, xData, {5});
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(Sigmoid, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // sigmoid(0) = 0.5
    std::vector<float> xData = {0.0f};
    std::vector<float> expected = {0.5f};

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Sigmoid, xData, {1});
    ASSERT_EQ(output.size(), expected.size());
    EXPECT_NEAR(output[0], expected[0], 1e-5f);
}

TEST(Silu, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // silu(x) = x * sigmoid(x)
    // silu(1) = 1 * 1/(1+exp(-1)) ≈ 0.7311
    std::vector<float> xData = {1.0f};
    float expected_silu = 1.0f / (1.0f + std::exp(-1.0f)); // ≈ 0.7311

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Silu, xData, {1});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output[0], expected_silu, 1e-4f);
}

TEST(Gelu, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
    // gelu(1) ≈ 0.8413
    std::vector<float> xData = {1.0f};
    float expected_gelu = 1.0f * 0.5f * (1.0f + std::erf(1.0f / std::sqrt(2.0f)));

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Gelu, xData, {1});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output[0], expected_gelu, 1e-4f);
}

TEST(Softplus, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // softplus(x) = log(1 + exp(x))
    // softplus(0) = log(2) ≈ 0.6931
    std::vector<float> xData = {0.0f};
    float expected_softplus = std::log(1.0f + std::exp(0.0f)); // log(2)

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Softplus, xData, {1});
    ASSERT_EQ(output.size(), 1u);
    EXPECT_NEAR(output[0], expected_softplus, 1e-5f);
}

TEST(Tanh, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // tanh(0) = 0, tanh(1) ≈ 0.7616
    std::vector<float> xData = {0.0f, 1.0f};
    std::vector<float> expected = {0.0f, std::tanh(1.0f)};

    auto output = runUnaryOnDevice(INFINI_DEVICE_CPU, 0, OpType::Tanh, xData, {2});
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }
}

// ---- Multi-platform tests (guarded by compile-time flags) ----

#ifdef USE_CUDA
TEST(Relu, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Relu, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Relu, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Relu, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
TEST(Sigmoid, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Sigmoid, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Sigmoid, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Sigmoid, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
TEST(Silu, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Silu, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Silu, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Silu, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
TEST(Gelu, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Gelu, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Gelu, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Gelu, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
TEST(Softplus, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Softplus, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Softplus, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Softplus, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
TEST(Tanh, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadUnaryTest<float>(OpType::Tanh, INFINI_DEVICE_NVIDIA, 0,
                                   DataType(INFINI_DTYPE_F32), true);
}
TEST(Tanh, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadUnaryTest<uint16_t>(OpType::Tanh, INFINI_DEVICE_NVIDIA, 0,
                                      DataType(INFINI_DTYPE_F16));
}
#endif // USE_CUDA

#ifdef USE_METAX
TEST(Relu, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Relu, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Sigmoid, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Sigmoid, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Silu, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Silu, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Gelu, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Gelu, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Softplus, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Softplus, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Tanh, MultiThread_CPU_MetaX_F32) {
    runMultiThreadUnaryTest<float>(OpType::Tanh, INFINI_DEVICE_METAX, 0,
                                   DataType(INFINI_DTYPE_F32));
}
#endif // USE_METAX

#ifdef USE_ILUVATAR
TEST(Relu, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Relu, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Sigmoid, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Sigmoid, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Silu, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Silu, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Gelu, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Gelu, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Softplus, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Softplus, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Tanh, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadUnaryTest<float>(OpType::Tanh, INFINI_DEVICE_ILUVATAR, 0,
                                   DataType(INFINI_DTYPE_F32));
}
#endif // USE_ILUVATAR

#ifdef USE_MOORE
TEST(Relu, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Relu, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Sigmoid, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Sigmoid, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Silu, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Silu, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Gelu, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Gelu, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Softplus, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Softplus, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
TEST(Tanh, MultiThread_CPU_Moore_F32) {
    runMultiThreadUnaryTest<float>(OpType::Tanh, INFINI_DEVICE_MOORE, 0,
                                   DataType(INFINI_DTYPE_F32));
}
#endif // USE_MOORE

} // namespace infini
