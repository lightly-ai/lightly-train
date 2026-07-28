find_package(TensorRT REQUIRED)

add_executable(od_infer_tensorrt main.cpp)

target_include_directories(od_infer_tensorrt PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(od_infer_tensorrt PRIVATE
  TensorRT::nvinfer
  ${OpenCV_LIBS}
  CUDA::cudart
)
