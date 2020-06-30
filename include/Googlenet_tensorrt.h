//
// Created by zf on 2020/4/13.
//

#ifndef CAFFE_GOOGLENET_TENSORRT_H
#define CAFFE_GOOGLENET_TENSORRT_H

#include <string>
#include <opencv4/opencv2/core/core.hpp>

using namespace std;

struct PersonDetectResultObeject {
    float ClassProb;
    std::string ClassName;
};

int scanFiles(vector<string> &fileList, string inputDirectory);

struct PersonDetectResultObeject mobilenet_detect(cv::Mat &rgbImage);



#endif //CAFFE_MOBILENET_TENSORRT_H
