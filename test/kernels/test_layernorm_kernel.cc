#include "core/runtime.h"
#include "operators/LayerNorm.h"
#include "utils/test_utils.h"
#include "gtest/gtest.h"
#include <cmath>
#include <thread>

namespace infini {

template <typename T> struct LayerNormThreadParams {
    infiniDevice_t device = INFINI_DEVICE_CPU;
    int deviceId = 0;
    Shape shapeX, shapeW;
    DataType dataType = DataType(INFINI_DTYPE_F32);
    std::vector<T> inputXData, inputWData, inputBData, outputData;
    float eps = 1e-5f;
    bool completed = false;
};

template <typename T>
void layernormDeviceThreadFunc(LayerNormThreadParams<T> &params) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(params.device, params.deviceId);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor(params.shapeX, params.dataType);
    auto W = g->addTensor(params.shapeW, params.dataType);
    auto B = g->addTensor(params.shapeW, params.dataType);
    auto op = g->addOp<LayerNormObj>(X, W, B, nullptr, params.eps);

    X->setData(params.inputXData.data());
    W->setData(params.inputWData.data());
    B->setData(params.inputBData.data());
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
void runMultiThreadLayerNormTest(infiniDevice_t targetDevice, int targetId,
                                  const DataType &dataType, bool print = false) {
    Shape shapeX = {2, 8};
    Shape shapeW = {8};
    size_t numX = 16, numW = 8;

    auto xData = generateRandomData<T>(numX, static_cast<T>(-2), static_cast<T>(2));
    std::vector<T> wData(numW, static_cast<T>(1));
    std::vector<T> bData(numW, static_cast<T>(0));

    LayerNormThreadParams<T> cpuParams, devParams;
    for (auto *p : {&cpuParams, &devParams}) {
        p->shapeX = shapeX;
        p->shapeW = shapeW;
        p->dataType = dataType;
        p->inputXData = xData;
        p->inputWData = wData;
        p->inputBData = bData;
        p->eps = 1e-5f;
    }
    cpuParams.device = INFINI_DEVICE_CPU;
    cpuParams.deviceId = 0;
    devParams.device = targetDevice;
    devParams.deviceId = targetId;

    std::thread cpuThread(layernormDeviceThreadFunc<T>, std::ref(cpuParams));
    std::thread devThread(layernormDeviceThreadFunc<T>, std::ref(devParams));
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
        std::cout << "LayerNorm: Errors=" << numErrors << ", MaxErr=" << maxError << std::endl;
    EXPECT_EQ(numErrors, 0u) << "CPU vs device mismatch (max error: " << maxError << ")";
}

// Reference LayerNorm for a single row of length n
static std::vector<float> layernormRef(const std::vector<float> &x,
                                       const std::vector<float> &w,
                                       const std::vector<float> &b,
                                       float eps) {
    size_t n = x.size();
    float mean = 0.0f;
    for (float v : x) mean += v;
    mean /= n;
    float var = 0.0f;
    for (float v : x) var += (v - mean) * (v - mean);
    var /= n;
    float invstd = 1.0f / std::sqrt(var + eps);
    std::vector<float> y(n);
    for (size_t i = 0; i < n; ++i)
        y[i] = (x[i] - mean) * invstd * w[i] + b[i];
    return y;
}

// ---- CPU absolute correctness ----

TEST(LayerNorm, CPU_F32_AbsoluteCorrectness) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    // x=[1.0, 2.0, 3.0], w=[1,1,1], b=[0,0,0], eps=1e-5
    std::vector<float> xData = {1.0f, 2.0f, 3.0f};
    std::vector<float> wData = {1.0f, 1.0f, 1.0f};
    std::vector<float> bData = {0.0f, 0.0f, 0.0f};
    auto expected = layernormRef(xData, wData, bData, 1e-5f);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({1, 3}, DataType(INFINI_DTYPE_F32));
    auto W = g->addTensor({3}, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor({3}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);
    X->setData(xData.data());
    W->setData(wData.data());
    B->setData(bData.data());
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

TEST(LayerNorm, CPU_F32_SingleDevice) {
    RuntimeObj::init();
    Runtime &runtime = RuntimeObj::getInstance();
    runtime->initThreadContext(INFINI_DEVICE_CPU, 0);

    Graph g = make_ref<GraphObj>(runtime);
    auto X = g->addTensor({2, 4}, DataType(INFINI_DTYPE_F32));
    auto W = g->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto B = g->addTensor({4}, DataType(INFINI_DTYPE_F32));
    auto op = g->addOp<LayerNormObj>(X, W, B, nullptr, 1e-5f);

    std::vector<float> xData = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<float> wData = {1, 1, 1, 1};
    std::vector<float> bData = {0, 0, 0, 0};
    X->setData(xData.data());
    W->setData(wData.data());
    B->setData(bData.data());
    runtime->dataMalloc(g);
    runtime->run(g);
    std::cout << "CPU LayerNorm Output:" << std::endl;
    op->getOutput(0)->printData(runtime);
}

// ---- Multi-platform tests ----

#ifdef USE_CUDA
TEST(LayerNorm, MultiThread_CPU_NVIDIA_F32) {
    runMultiThreadLayerNormTest<float>(INFINI_DEVICE_NVIDIA, 0,
                                       DataType(INFINI_DTYPE_F32), true);
}
TEST(LayerNorm, MultiThread_CPU_NVIDIA_F16) {
    runMultiThreadLayerNormTest<uint16_t>(INFINI_DEVICE_NVIDIA, 0,
                                          DataType(INFINI_DTYPE_F16));
}
#endif

#ifdef USE_METAX
TEST(LayerNorm, MultiThread_CPU_MetaX_F32) {
    runMultiThreadLayerNormTest<float>(INFINI_DEVICE_METAX, 0,
                                       DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_ILUVATAR
TEST(LayerNorm, MultiThread_CPU_Iluvatar_F32) {
    runMultiThreadLayerNormTest<float>(INFINI_DEVICE_ILUVATAR, 0,
                                       DataType(INFINI_DTYPE_F32));
}
#endif

#ifdef USE_MOORE
TEST(LayerNorm, MultiThread_CPU_Moore_F32) {
    runMultiThreadLayerNormTest<float>(INFINI_DEVICE_MOORE, 0,
                                       DataType(INFINI_DTYPE_F32));
}
#endif

} // namespace infini
