//
// Copyright (c) Lightly AG and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Recipe: run LT-DETR object detection inference with a serialized TensorRT
// engine, allocating inputs/outputs directly on the GPU (zero-copy).
//
// This is a single-purpose example, not a general-purpose CLI tool: edit the
// constants below to point at your own exported engine/image, then rebuild.
//
// TensorRT engines carry no metadata (unlike ONNX's metadata_props), so
// normalization/image size/class names must be hardcoded here -- this
// mirrors lightly_train's own benchmark harness
// (src/lightly_train/_commands/benchmark_backends.py::TensorRTBackend),
// which is the reference this file ports to C++.
//
// Export the engine first, on a machine with lightly-train + tensorrt-cu12
// installed:
//
//   import lightly_train
//   model = lightly_train.load_model("ltdetrv2-s-coco")
//   print(model.image_size, model.image_normalize, model.classes)
//   model.export_tensorrt("model.trt")
//
// The printed image_size/image_normalize/classes must match the constants
// below -- update them if you export a different checkpoint.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/imgcodecs.hpp>

#include "../common/detection_utils.hpp"

namespace {

// --- Edit these to match your exported engine ---
constexpr const char* kEnginePath = "model.trt";
constexpr const char* kImagePath = "image.jpg";
constexpr const char* kOutputPath = "output.jpg";
constexpr float kThreshold = 0.6f;
constexpr int kDeviceId = 0;
constexpr int kModelHeight = 640;  // must match model.image_size[0] at export time
constexpr int kModelWidth = 640;   // must match model.image_size[1] at export time
// Must match the exported model's postprocessor config (LT-DETR "Generic"
// default is 300, see src/lightly_train/_task_models/ltdetr_object_detection/config.py).
constexpr int kNumTopQueries = 300;

// TensorRT engines carry no normalization metadata (unlike ONNX), so this
// must be hardcoded to match model.image_normalize for the exported
// checkpoint. "ltdetrv2-s-coco" uses ImageNet statistics; other checkpoints
// may use LT-DETR's generic default of mean=(0,0,0), std=(1,1,1) (i.e. only
// the /255 scaling applies) -- always cross-check against the printed value
// from your own export.
const od_common::ImageNormalize kNormalize = {{0.485f, 0.456f, 0.406f},
                                               {0.229f, 0.224f, 0.225f}};

// Must match model.classes (in internal-index order) for the exported
// checkpoint -- print(model.classes) in Python and paste its values here in
// order. The list below is the standard 80-class COCO ordering used by
// "ltdetrv2-s-coco".
const std::vector<std::string> kClassNames = {
    "person",        "bicycle",      "car",           "motorcycle",
    "airplane",      "bus",          "train",         "truck",
    "boat",          "traffic light","fire hydrant",  "stop sign",
    "parking meter", "bench",        "bird",          "cat",
    "dog",           "horse",        "sheep",         "cow",
    "elephant",      "bear",         "zebra",         "giraffe",
    "backpack",      "umbrella",     "handbag",       "tie",
    "suitcase",      "frisbee",      "skis",          "snowboard",
    "sports ball",   "kite",         "baseball bat",  "baseball glove",
    "skateboard",    "surfboard",    "tennis racket", "bottle",
    "wine glass",    "cup",          "fork",          "knife",
    "spoon",         "bowl",         "banana",        "apple",
    "sandwich",      "orange",       "broccoli",      "carrot",
    "hot dog",       "pizza",        "donut",         "cake",
    "chair",         "couch",        "potted plant",  "bed",
    "dining table",  "toilet",       "tv",            "laptop",
    "mouse",         "remote",       "keyboard",      "cell phone",
    "microwave",     "oven",         "toaster",       "sink",
    "refrigerator",  "book",         "clock",         "vase",
    "scissors",      "teddy bear",   "hair drier",    "toothbrush",
};

void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << msg << std::endl;
    }
  }
};

std::vector<char> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open engine file: " + path);
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<char> buffer(static_cast<size_t>(size));
  if (!file.read(buffer.data(), size)) {
    throw std::runtime_error("Failed to read engine file: " + path);
  }
  return buffer;
}

size_t element_size(nvinfer1::DataType dtype) {
  switch (dtype) {
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    case nvinfer1::DataType::kINT64:
      return 8;
    case nvinfer1::DataType::kINT32:
      return 4;
    default:
      throw std::runtime_error("Unsupported TensorRT tensor dtype.");
  }
}

}  // namespace

int main() {
  check_cuda(cudaSetDevice(kDeviceId), "cudaSetDevice");

  Logger logger;
  std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));

  const std::vector<char> engine_data = read_file(kEnginePath);
  std::unique_ptr<nvinfer1::ICudaEngine> engine(
      runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine) {
    throw std::runtime_error("Failed to deserialize TensorRT engine.");
  }

  std::unique_ptr<nvinfer1::IExecutionContext> context(engine->createExecutionContext());
  if (!context) {
    throw std::runtime_error("Failed to create TensorRT execution context.");
  }

  // Derive input/output tensor names from the engine (TensorRT 10.x API --
  // note the C++ naming differs from Python's get_tensor_name/get_tensor_mode).
  std::string input_name;
  std::vector<std::string> output_names;
  for (int i = 0; i < engine->getNbIOTensors(); ++i) {
    const char* name = engine->getIOTensorName(i);
    if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name = name;
    } else {
      output_names.emplace_back(name);
    }
  }
  if (input_name.empty()) {
    throw std::runtime_error("Engine has no input tensor.");
  }

  const auto pre = od_common::preprocess_image(kImagePath, kModelWidth, kModelHeight,
                                                kNormalize);

  // Single-image recipe -- batch size is always 1.
  context->setInputShape(input_name.c_str(), nvinfer1::Dims4{1, 3, kModelHeight, kModelWidth});

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

  float* d_input = nullptr;
  const size_t input_count = pre.data.size();
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), input_count * sizeof(float)),
             "cudaMalloc(d_input)");
  check_cuda(cudaMemcpyAsync(d_input, pre.data.data(), input_count * sizeof(float),
                              cudaMemcpyHostToDevice, stream),
             "cudaMemcpyAsync(d_input)");
  context->setTensorAddress(input_name.c_str(), d_input);

  struct OutputBuffer {
    void* device_ptr = nullptr;
    size_t num_elements = 0;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
  };
  std::vector<std::pair<std::string, OutputBuffer>> outputs;
  for (const auto& name : output_names) {
    // Output shapes depend on the batch dim set via setInputShape() above,
    // so they must be read from the execution context (resolved shape), not
    // the engine (which may still report -1 for the dynamic batch dim).
    const nvinfer1::Dims out_dims = context->getTensorShape(name.c_str());
    size_t num_elements = 1;
    for (int d = 0; d < out_dims.nbDims; ++d) num_elements *= out_dims.d[d];

    const nvinfer1::DataType dtype = engine->getTensorDataType(name.c_str());
    OutputBuffer buffer;
    buffer.num_elements = num_elements;
    buffer.dtype = dtype;
    check_cuda(cudaMalloc(&buffer.device_ptr, num_elements * element_size(dtype)),
               "cudaMalloc(output)");
    context->setTensorAddress(name.c_str(), buffer.device_ptr);
    outputs.emplace_back(name, buffer);
  }

  if (!context->enqueueV3(stream)) {
    throw std::runtime_error("TensorRT enqueueV3 failed.");
  }
  check_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");

  const OutputBuffer* logits_buffer = nullptr;
  const OutputBuffer* boxes_buffer = nullptr;
  for (const auto& [name, buffer] : outputs) {
    if (name == "logits") {
      logits_buffer = &buffer;
    } else if (name == "boxes") {
      boxes_buffer = &buffer;
    }
  }
  if (logits_buffer == nullptr || boxes_buffer == nullptr) {
    throw std::runtime_error("Engine did not produce the expected 'logits'/'boxes' outputs.");
  }
  if (logits_buffer->dtype != nvinfer1::DataType::kFLOAT ||
      boxes_buffer->dtype != nvinfer1::DataType::kFLOAT) {
    throw std::runtime_error(
        "This recipe assumes fp32 outputs; export the engine with precision=\"fp32\".");
  }

  // logits: (1, num_queries, num_classes), boxes: (1, num_queries, 4).
  const int num_queries = static_cast<int>(boxes_buffer->num_elements / 4);
  const int num_classes = static_cast<int>(logits_buffer->num_elements / num_queries);

  std::vector<float> logits_host(logits_buffer->num_elements);
  std::vector<float> boxes_host(boxes_buffer->num_elements);
  check_cuda(cudaMemcpy(logits_host.data(), logits_buffer->device_ptr,
                         logits_host.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(logits_host)");
  check_cuda(cudaMemcpy(boxes_host.data(), boxes_buffer->device_ptr,
                         boxes_host.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(boxes_host)");

  const auto detections =
      od_common::postprocess(logits_host.data(), boxes_host.data(), num_queries,
                              num_classes, kNumTopQueries, pre.orig_w, pre.orig_h,
                              kThreshold);

  std::cout << "Found " << detections.size() << " detection(s):\n";
  for (const auto& det : detections) {
    const std::string name = (static_cast<size_t>(det.label) < kClassNames.size())
                                  ? kClassNames[static_cast<size_t>(det.label)]
                                  : std::to_string(det.label);
    std::printf("  %-16s score=%.3f box=(%.1f, %.1f, %.1f, %.1f)\n", name.c_str(),
                det.score, det.x1, det.y1, det.x2, det.y2);
  }

  cv::Mat image = cv::imread(kImagePath, cv::IMREAD_COLOR);
  od_common::draw_detections(image, detections, kClassNames);
  cv::imwrite(kOutputPath, image);
  std::cout << "Wrote " << kOutputPath << std::endl;

  cudaFree(d_input);
  for (const auto& [name, buffer] : outputs) {
    cudaFree(buffer.device_ptr);
  }
  cudaStreamDestroy(stream);
  return 0;
}
