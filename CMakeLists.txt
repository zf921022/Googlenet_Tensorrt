cmake_minimum_required(VERSION 3.10)
project(GooglenetTensorrt)
set(CMAKE_CXX_STANDARD 14)


include_directories(/usr/local/cuda/include)
include_directories(/usr/local/include/opencv4)
include_directories(./include/common)
include_directories(./include)

link_directories(/usr/local/cuda/lib64)
link_directories(/usr/local/lib)
link_directories(/home/zf/TensorRT-6.0.1.5/lib)
link_directories(${OpenCV_LIBARIES})

aux_source_directory(src DIR_SRCS)
add_executable(GooglenetTensorrt ${DIR_SRCS})

target_link_libraries(GooglenetTensorrt cudart)
target_link_libraries(GooglenetTensorrt nvinfer)
target_link_libraries(GooglenetTensorrt nvcaffe_parser)
target_link_libraries(GooglenetTensorrt opencv_core)
target_link_libraries(GooglenetTensorrt opencv_highgui)
target_link_libraries(GooglenetTensorrt opencv_imgcodecs)
target_link_libraries(GooglenetTensorrt opencv_imgproc)
#target_link_libraries(MobilenetTensorrt ${OpenCV_LIBS})
