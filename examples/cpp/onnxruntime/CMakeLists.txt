# Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-gpu-<version> -- the
# official prebuilt GPU tarball from
# https://github.com/microsoft/onnxruntime/releases. The pip `onnxruntime-gpu`
# wheel does not reliably ship the C++ headers/import library needed here.
find_path(ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES include
)
find_library(ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT} ENV ONNXRUNTIME_ROOT
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRUNTIME
  DEFAULT_MSG
  ONNXRUNTIME_LIBRARY
  ONNXRUNTIME_INCLUDE_DIR
)

add_executable(od_infer_onnxruntime main.cpp)

target_include_directories(od_infer_onnxruntime PRIVATE
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(od_infer_onnxruntime PRIVATE
  ${ONNXRUNTIME_LIBRARY}
  ${OpenCV_LIBS}
  CUDA::cudart
)
