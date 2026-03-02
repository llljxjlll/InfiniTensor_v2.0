#include "core/runtime.h"
#include "operators/Conv.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <thread>

namespace infini {

template <typename T> struct ConvThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeX, shapeW, shapeY;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputXData, inputWData, outputData;
    std::vector<int64_t> pads, strides, dilations;
    bool completed = false;
    std::string deviceName;
};

template <typename T> void convDeviceThreadFunc(ConvThreadParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(params.device, params.deviceId);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor(params.shapeX, params.dataType);
    auto W = g->addTensor(params.shapeW, params.dataType);
    auto op = g->addOp<ConvObj>(X, W, nullptr, nullptr, params.pads,
                                 params.strides, params.dilations);

    X->setData(params.inputXData.data());
    W->setData(params.inputWData.data());
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
}

template <typename T>
void runMultiThreadConvTest(infiniDevice_t targetDevice, int targetId,
                            const DataType &dataType, bool print = false) {
    // x=[1,1,4,4], w=[1,1,2,2], pads=[0,0], strides=[1,1], dilations=[1,1]
    Shape shapeX = {1, 1, 4, 4};
    Shape shapeW = {1, 1, 2, 2};
    size_t numX = 16, numW = 4;

    auto xData = generateRandomData<T>(numX, static_cast<T>(-1), static_cast<T>(1));
    auto wData = generateRandomData<T>(numW, static_cast<T>(-1), static_cast<T>(1));

    ConvThreadParams<T> cpuParams, devParams;
    for (auto *p : {&cpuParams, &devParams}) {
        p->shapeX = shapeX;
        p->shapeW = shapeW;
        p->shapeY = {1, 1, 3, 3};
        p->dataType = dataType;
        p->inputXData = xData;
        p->inputWData = wData;
        p->pads = {0, 0};
        p->strides = {1, 1};
        p->dilations = {1, 1};
    }
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    cpuParams.deviceName = "CPU";
    devParams.device = targetDevice;
    devParams.deviceId = targetId;

    std::thread cpuThread(convDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(convDeviceThreadFunc<T>, std::ref(devParams));
    cpuThread.join();
    devThread.join();

    ASSERT_TRUE(cpuParams.completed);
    ASSERT_TRUE(devParams.completed);
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
        std::cout << "Conv: Errors=" << numErrors << ", MaxErr=" << maxError << std::endl;
    EXPECT_EQ(numErrors, 0u) << "CPU vs device mismatch (max error: " << maxError << ")";
}

// ---- CPU absolute correctness ----

TEST(Conv, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // x=[1,1,3,3]: [[1,2,3],[4,5,6],[7,8,9]]
    // w=[1,1,2,2]: [[1,0],[0,1]]
    // pads=[0,0], strides=[1,1], dilations=[1,1]
    // y[0][0] = 1+5 = 6, y[0][1] = 2+6 = 8, y[1][0] = 4+8 = 12, y[1][1] = 5+9 = 14
    std::vector<float> xData = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<float> wData = {1, 0, 0, 1};
    std::vector<float> expected = {6, 8, 12, 14};

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({1, 1, 3, 3}, DataType(INFINI_DTYPE_F32));
    auto W = g->addTensor({1, 1, 2, 2}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(X, W, nullptr, nullptr,
                                 std::vector<int64_t>{0, 0},
                                 std::vector<int64_t>{1, 1},
                                 std::vector<int64_t>{1, 1});
    X->setData(xData.data());
    W->setData(wData.data());
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

TEST(Conv, CPU_F32_SingleDevice) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({1, 1, 4, 4}, DataType(INFINI_DTYPE_F32));
    auto W = g->addTensor({1, 1, 2, 2}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<ConvObj>(X, W, nullptr, nullptr,
                                 std::vector<int64_t>{0, 0},
                                 std::vector<int64_t>{1, 1},
                                 std::vector<int64_t>{1, 1});
    std::vector<float> xData(16);
    for (int i = 0; i < 16; ++i)
        xData[i] = static_cast<float>(i + 1);
    std::vector<float> wData = {1.0f, 0.0f, 0.0f, 1.0f};
    X->setData(xData.data());
    W->setData(wData.data());
    runtime->dataMalloc(g);
    runtime->run(g);
    std::cout << "CPU Conv Output:" << std::endl;
    op->getOutput(0)->printData(runtime);
}

// ---- Multi-platform tests ----

#ifdef USE_CUDA
TEST(Conv, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadConvTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                  DataType(INFINI_DTYPE_F32), true);
}
TEST(Conv, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadConvTest<uint16_t>(INFINI_DEVICE_NVIDIA, 0,
                                     DataType(INFINI_DTYPE_F16));
}
#endif

#ifdef USE_METAX
TEST(Conv, MultiThread_CPU_MetaX_F32) {
    runMultiThreadConvTest<float>(INFINI_DEVICE_METAX, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_ILUVATAR
TEST(Conv, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadConvTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_MOORE
TEST(Conv, MultiThread_CPU_Moore_F32) {
    runMultiThreadConvTest<float>(INFINI_DEVICE_MOORE, 0,
                                  DataType(INFINI_DTYPE_F32));
}
#endif

} // namespace infini
