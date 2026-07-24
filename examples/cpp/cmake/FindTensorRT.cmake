# Locates TensorRT for the `tensorrt` recipe below.
#
# Most TensorRT installs -- including the `tensorrt-cu12` pip wheel used by
# examples/notebooks/object_detection_export.ipynb -- do not ship an official
# TensorRTConfig.cmake, so this module finds the headers/libraries directly.
#
# Set -DTensorRT_ROOT=<path> to point at a TensorRT install, e.g. the
# site-packages directory of a pip-installed `tensorrt-cu12` package
# (`python -c "import tensorrt, os; print(os.path.dirname(tensorrt.__file__))"`)
# or an extracted TensorRT tarball/.deb install root.

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
