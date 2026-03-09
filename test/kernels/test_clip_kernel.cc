#include "core/runtime.h"
#include "operators/Clip.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <thread>

namespace infini {

// Per-device thread parameters for Clip
template <typename T> struct ClipThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeX;
    Shape shapeMin; // typically {1} for scalar bounds
    Shape shapeMax; // typically {1} for scalar bounds
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputXData;
    std::vector<T> inputMinData;
    std::vector<T> inputMaxData;
    std::vector<T> outputData;
    bool completed = false;
    std::string deviceName;
};

// Execute Clip on a specific device and collect output
template <typename T> void clipDeviceThreadFunc(ClipThreadParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(params.device, params.deviceId);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor(params.shapeX, params.dataType);
    auto Min = g->addTensor(params.shapeMin, params.dataType);
    auto Max = g->addTensor(params.shapeMax, params.dataType);
    auto op = g->addOp<ClipObj>(X, Min, Max, nullptr);

    X->setData(params.inputXData.data());
    Min->setData(params.inputMinData.data());
    Max->setData(params.inputMaxData.data());
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
}

// Run Clip on a single device and return output as float vector.
// Used for absolute correctness checks against known expected values.
std::vector<float> runClipOnDevice(infiniDevice_t device, int deviceId,
                                   const std::vector<float> &xData,
                                   float minVal, float maxVal,
                                   const Shape &shapeX) {
    ClipThreadParams<float> params;
    params.device = device;
    params.deviceId = deviceId;
    params.shapeX = shapeX;
    params.shapeMin = {1};
    params.shapeMax = {1};
    params.dataType = DataType(INFINI_DTYPE_F32);
    params.inputXData = xData;
    params.inputMinData = {minVal};
    params.inputMaxData = {maxVal};

    clipDeviceThreadFunc(params);
    return params.outputData;
}

// Run Clip on two devices in parallel and compare results.
template <typename T>
void runMultiThreadClipTest(infiniDevice_t targetDevice, int targetId,
                            const DataType &dataType, bool print = false) {
    Shape shapeX = {2, 3, 4};
    Shape shapeMin = {1};
    Shape shapeMax = {1};

    size_t numX = 1;
    for (auto d : shapeX)
        numX *= d;

    auto xData =
        generateRandomData<T>(numX, static_cast<T>(-10), static_cast<T>(10));
    // Use fixed min/max for reproducibility
    std::vector<T> minData = {static_cast<T>(-2)};
    std::vector<T> maxData = {static_cast<T>(5)};

    ClipThreadParams<T> cpuParams, devParams;

    // CPU thread
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.shapeX = shapeX;
    cpuParams.shapeMin = shapeMin;
    cpuParams.shapeMax = shapeMax;
    cpuParams.dataType = dataType;
    cpuParams.inputXData = xData;
    cpuParams.inputMinData = minData;
    cpuParams.inputMaxData = maxData;
    cpuParams.deviceName = "CPU";

    // Target device thread
    devParams.device = targetDevice;
    devParams.deviceId = targetId;
    devParams.shapeX = shapeX;
    devParams.shapeMin = shapeMin;
    devParams.shapeMax = shapeMax;
    devParams.dataType = dataType;
    devParams.inputXData = xData;
    devParams.inputMinData = minData;
    devParams.inputMaxData = maxData;

    if (print) {
        std::cout << "========================================" << std::endl;
        std::cout << "Running Multi-Thread Clip Test" << std::endl;
        std::cout << "DataType: " << dataType.toString() << std::endl;
        std::cout << "Shape X: " << vecToString(shapeX) << std::endl;
        std::cout << "========================================" << std::endl;
    }

    std::thread cpuThread(clipDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(clipDeviceThreadFunc<T>, std::ref(devParams));
    cpuThread.join();
    devThread.join();

    ASSERT_TRUE(cpuParams.completed) << "CPU thread failed";
    ASSERT_TRUE(devParams.completed) << "Device thread failed";
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
        std::cout << "========================================" << std::endl;
    }

    EXPECT_EQ(numErrors, 0u)
        << "Results mismatch between CPU and device (max error: " << maxError
        << ")";
}

// ---- CPU absolute correctness test (always enabled) ----

TEST(Clip, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // X = [-5, -1, 0, 3, 7], min = -2, max = 4
    // Expected: [-2, -1, 0, 3, 4]
    std::vector<float> xData = {-5.0f, -1.0f, 0.0f, 3.0f, 7.0f};
    std::vector<float> expected = {-2.0f, -1.0f, 0.0f, 3.0f, 4.0f};

    auto output =
        runClipOnDevice(INFINI_DEVICE_CPU, 0, xData, -2.0f, 4.0f, {5});
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(output[i], expected[i], 1e-5f) << "Mismatch at index " << i;
    }
}

TEST(Clip, CPU_F32_SingleDevice) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Shape shapeX = {3, 4};
    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor(shapeX, DataType(INFINI_DTYPE_F32));
    auto Min = g->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto Max = g->addTensor({1}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ClipObj>(X, Min, Max, nullptr);

    std::vector<float> xData(12);
    for (int i = 0; i < 12; ++i)
        xData[i] = static_cast<float>(i - 6);
    std::vector<float> minData = {-3.0f};
    std::vector<float> maxData = {3.0f};

    X->setData(xData.data());
    Min->setData(minData.data());
    Max->setData(maxData.data());
    runtime->dataMalloc(g);
    runtime->run(g);

    std::cout << "CPU Clip Output: " << std::endl;
    op->getOutput(0)->printData(runtime);
}

// ---- Multi-platform tests (guarded by compile-time flags) ----

#ifdef USE_CUDA
TEST(Clip, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadClipTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                  DataType(INFINI_DTYPE_F32), true);
}

TEST(Clip, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadClipTest<uint16_t>(INFINI_DEVICE_NVIDIA, 0,
                                     DataType(INFINI_DTYPE_F16));
}
#endif // USE_CUDA

#ifdef USE_METAX
TEST(Clip, MultiThread_CPU_MetaX_F32) {
    runMultiThreadClipTest<float>(INFINI_DEVICE_METAX, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif // USE_METAX

#ifdef USE_ILUVATAR
TEST(Clip, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadClipTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif // USE_ILUVATAR

#ifdef USE_MOORE
TEST(Clip, MultiThread_CPU_Moore_F32) {
    runMultiThreadClipTest<float>(INFINI_DEVICE_MOORE, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif // USE_MOORE

} // namespace infini
