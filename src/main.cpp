//
// Created by zf on 2020/4/13.
//

/*
    sample code about converting googlenet caffemodel to trt
*/

#include <string>
#include "Googlenet_tensorrt.h"
#include <opencv4/opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;


std::string img_dir = "../images/";


// main
int main(int argc, char** argv)
{
    vector<string> files;
    size_t size_files;
    size_files = scanFiles(files,img_dir);
    string test_image;
    for(int t= 0; t<size_files;t++) {
        test_image = files[t];
        cv::Mat test_image_Data = cv::imread(img_dir+test_image);
        PersonDetectResultObeject DetectedResult = mobilenet_detect(test_image_Data);
    }
    return 0;
}