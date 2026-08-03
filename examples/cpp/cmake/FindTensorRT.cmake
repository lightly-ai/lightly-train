# Locates TensorRT for the `tensorrt` recipe below.
#
# TensorRT installs do not ship an official TensorRTConfig.cmake, so this
# module finds the headers/libraries directly. This requires a native
# TensorRT install with C++ headers (e.g. `libnvinfer-dev` via apt, or an
# extracted TensorRT tarball) -- the `tensorrt-cu12` pip wheel used by
# examples/notebooks/object_detection_export.ipynb does not ship `NvInfer.h`
# and cannot be used here.
#
# Set -DTensorRT_ROOT=<path> to point at a TensorRT tarball install root. Not
# needed for an apt install, since headers/libs are already on the default
# search path.

find_path(TensorRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES include
)

find_library(TensorRT_LIBRARY
  NAMES nvinfer
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64
)

find_library(TensorRT_ONNXPARSER_LIBRARY
  NAMES nvonnxparser
  HINTS ${TensorRT_ROOT} ENV TensorRT_ROOT
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
  DEFAULT_MSG
  TensorRT_LIBRARY
  TensorRT_INCLUDE_DIR
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
  add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
  set_target_properties(TensorRT::nvinfer PROPERTIES
    IMPORTED_LOCATION "${TensorRT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}"
  )
endif()

mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY TensorRT_ONNXPARSER_LIBRARY)
