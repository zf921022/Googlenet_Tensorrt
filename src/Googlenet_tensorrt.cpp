//
// Created by zf on 2020/4/13.
//

#include <assert.h>
#include <fstream>
#include <string>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <cuda_runtime_api.h>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"
#include "Googlenet_tensorrt.h"

#include <opencv2/opencv.hpp>

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace cv;
using namespace std;

static const int INPUT_C = 3;
static const int INPUT_H = 224;
static const int INPUT_W = 224;
static const int OUTPUT_SIZE = 2;

Logger gLogger;

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";


std::string MODEL_PROTOTXT = "../data/deploy.prototxt";
std::string CAFFE_MODEL = "../data/snapshot_iter_18338.caffemodel";

float fMeanBGR[] = {93.85, 101.09, 109.46};

// caffe to GIR model
void caffeToGIEModel(const std::string& deployFile,             // name for caffe prototxt
                     const std::string& modelFile,              // name for model
                     const std::vector<std::string>& outputs,   // network outputs
                     unsigned int maxBatchSize,                 // batch size - NB must be at least as large as the batch we want to run with)
                     IHostMemory *&gieModelStream)    // output buffer for the GIE model
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger);
    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();
    const IBlobNameToTensor* blobNameToTensor
            = parser->parse(deployFile.c_str(), modelFile.c_str(),
                            *network, nvinfer1::DataType::kFLOAT);

    // specify which tensors are outputs
    for (auto& s : outputs)
        network->markOutput(*blobNameToTensor->find(s.c_str()));

    // Build the engineI
    // builder->setHalf2Mode(true);
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(1 << 20);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    assert(engine);

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    // serialize the engine, then close everything down
    gieModelStream = engine->serialize();

    // 设置保存文件的名称为cached_model.bin
    std::string cache_path = "../engine/googlenet_tensorrt_person.engine";
    std::ofstream serialize_output_stream;

    // 将序列化的模型结果拷贝至serialize_str字符串
    std::string serialize_str;
    serialize_str.resize( gieModelStream->size() );
    memcpy((void*)serialize_str.data(), gieModelStream->data(), gieModelStream->size());

    // 将serialize_str字符串的内容输出至cached_path
    serialize_output_stream.open(cache_path);
    serialize_output_stream << serialize_str;
    serialize_output_stream.close();

    engine->destroy();
    builder->destroy();
    shutdownProtobufLibrary();
}


int getImgData(cv::Mat &ImageData, float *fData) {
    if(ImageData.empty()) return -1;
    resize(ImageData, ImageData, Size(INPUT_W, INPUT_H));
    int nr = ImageData.rows, nc = ImageData.cols, nChannels = ImageData.channels();
    if(INPUT_C != nChannels) {
        cout << "INPUT_C != nChannels" << endl;
        cout << "INPUT_C = "  << INPUT_C << ", nChannels = " << nChannels << endl;
        return -1;
    }
    switch (nChannels) {
        case 1:
            break;
        case 3:
            Vec3b* pr;
            for(int i = 0; i < nr; ++i)
            {
                pr = ImageData.ptr<Vec3b>(i);
                for (int j = 0; j < nc; ++j)
                {
                    for(int ic = 0; ic < nChannels; ++ic) {
                        fData[nr * nc * ic + i * nc + j] = float(pr[j][ic]) - fMeanBGR[ic];
                        // fData[kkk++] = float(pr[j][ic]) - fMeanBGR[ic];
                    }
                }
            }
            break;
        default:
            return -1;
    }
    return 1;
}

// inference
void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
            outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * INPUT_C * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE*sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//load engine from engineFilePath
static ICudaEngine *createEngine(std::string engineFilePath) {
    ICudaEngine *engine;
    std::vector<char> trtModelStream;
    size_t size{0};
    clock_t start, end;
    double dur;
    std::ifstream file(engineFilePath, std::ios::binary);
    if (file.good()) {
        file.seekg(0, std::ifstream::end);
        size = file.tellg(); //return size of file;
        file.seekg(0, std::ifstream::beg);
        trtModelStream.resize(size);
        file.read(trtModelStream.data(), size);
        file.close();

        IRuntime *infer = createInferRuntime(gLogger);
        engine = infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr);
        std::cout << "Engine File: " << engineFilePath << " has been successfully loaded." << std::endl;
        infer->destroy();
        return engine;
    }
    if(!file){
        std::cout << "Engine File: " << engineFilePath << " loading failed." << std::endl;
        // create a GIE model from the caffe model and serialize it to a stream
        cout << "convert caffe model ..." << endl;
        start = clock();
        IHostMemory *gieModelStream{nullptr};
        caffeToGIEModel(MODEL_PROTOTXT.c_str(), CAFFE_MODEL.c_str(),
                        std::vector <std::string> { OUTPUT_BLOB_NAME }, 1, gieModelStream);
        end = clock();
        dur = (double)(end - start);
        cout << "\t" << "time = " << (dur / CLOCKS_PER_SEC) << " s" << endl;
        IRuntime *runtime = createInferRuntime(gLogger);
        ICudaEngine *engine = runtime->deserializeCudaEngine(gieModelStream->data(), gieModelStream->size(), nullptr);
        if (gieModelStream) gieModelStream->destroy();
        runtime->destroy();
        return engine;
    }
}


#include <dirent.h>

/**
    Linux下扫描文件夹， 获得文件夹下的文件名
*/
int scanFiles(vector<string> &fileList, string inputDirectory)
{
    inputDirectory = inputDirectory.append("/");

    DIR *p_dir;
    const char* str = inputDirectory.c_str();

    p_dir = opendir(str);
    if( p_dir == NULL)
    {
        cout<< "can't open :" << inputDirectory << endl;
    }

    struct dirent *p_dirent;

    while ( p_dirent = readdir(p_dir))
    {
        string tmpFileName = p_dirent->d_name;
        if( tmpFileName == "." || tmpFileName == "..")
        {
            continue;
        }
        else
        {
            fileList.push_back(tmpFileName);
        }
    }
    closedir(p_dir);
    return fileList.size();
}

// main
PersonDetectResultObeject mobilenet_detect(cv::Mat &rgbImage)
{
    PersonDetectResultObeject DetectedResult;
    clock_t start, end;
    double dur;

    ifstream infile("../data/labels.txt");
    assert(infile);
    vector<string> label_names;
    string s;
    while (getline(infile, s)) {
        label_names.push_back(s);
        s.clear();
    }
    infile.close();

    cout << "read image ..." << endl;
    start = clock();
    float fData[INPUT_W * INPUT_H * INPUT_C];
    int nGetImage = getImgData(rgbImage, fData);
    if (nGetImage != 1) {
        cout << "read image error" << endl;
    }
    end = clock();
    dur = (double) (end - start);
    cout << "\t" << "time = " << (dur / CLOCKS_PER_SEC * 1000) << " ms" << endl;

    // deserialize the engine
    cout << "create runtime and engine..." << endl;

    ICudaEngine *engine;
    if(!engine){
    engine = createEngine("../engine/googlenet_tensorrt_person.engine");}
    IExecutionContext *context = engine->createExecutionContext();

    // run inference

    cout << "inference ..." << endl;
    float prob[OUTPUT_SIZE];
    start = clock();
    doInference(*context, fData, prob, 1);
    end = clock();
    dur = (double) (end - start);
    cout << "\t" << "time = " << (dur / CLOCKS_PER_SEC * 1000) << " ms" << endl;

    context->destroy();
//    engine->destroy();
//    runtime->destroy();
    cout << endl << "output result ..." << endl;

    float val{0.0f};
    int idx{0};
    for (unsigned int i = 0; i < OUTPUT_SIZE; i++) {
        if (prob[i] > val) {
            val = prob[i];
            idx = i;
        }
    }
    cout << "pred class = " << label_names[idx] << " " << idx << ", pred prob = " << val << endl;
    std::cout << std::endl;
    cv::namedWindow("test result", 0);
    cv::putText(rgbImage, label_names[idx], Point(rgbImage.cols/8, rgbImage.rows/2),
                FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2);
    cv::imshow("test result",rgbImage);
    cv::waitKey(0);
    cout << "Done" << endl;
    DetectedResult.ClassName = label_names[idx];
    DetectedResult.ClassProb = val;
    //
    return DetectedResult;
}
