cmake_minimum_required(VERSION 3.18)
project(lightly_train_cpp_inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(CUDAToolkit REQUIRED)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_subdirectory(onnxruntime)
add_subdirectory(tensorrt)
