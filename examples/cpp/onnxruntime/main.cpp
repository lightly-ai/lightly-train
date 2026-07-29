//
// Copyright (c) Lightly AG and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Recipe: run LT-DETR object detection inference with ONNX Runtime's CUDA
// execution provider, allocating the model input and outputs directly on
// the GPU (zero-copy) via Ort::IoBinding, without relying on ONNX Runtime's
// own CUDA allocator.
//
// This is a single-purpose example, not a general-purpose CLI tool: edit the
// constants below to point at your own exported model/image, then rebuild.
//
// Export the model first, on a machine with lightly-train installed:
//
//   import lightly_train
//   model = lightly_train.load_model("dinov3/vitt16-ltdetr-coco")
//   print(model.image_size, model.image_normalize, model.classes)
//   model.export_onnx("model.onnx")
//
// The printed image_size/image_normalize/classes must match the constants
// below -- update them if you export a different checkpoint.
#include <array>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/imgcodecs.hpp>

#include "../common/detection_utils.hpp"

namespace {

// --- Edit these to match your exported model ---
constexpr const char* kModelPath = "model.onnx";
constexpr const char* kImagePath = "image.jpg";
constexpr const char* kOutputPath = "output.jpg";
constexpr float kThreshold = 0.6f;
constexpr int kDeviceId = 0;
constexpr int kModelHeight = 640;  // must match model.image_size[0] at export time
constexpr int kModelWidth = 640;   // must match model.image_size[1] at export time
// Must match the exported model's postprocessor config (LT-DETR "Generic"
// default is 300, see src/lightly_train/_task_models/ltdetr_object_detection/config.py).
constexpr int kNumTopQueries = 300;
// Must match the exported model's raw decoder query count (RTDETRv2/D-FINE
// transformer "num_queries" config, default 300, same config.py as above).
// This is the model's fixed "logits"/"boxes" output size, not to be confused
// with kNumTopQueries above (the postprocessor's independent top-K selection
// count) -- they share the same default but can differ. If unsure, inspect
// the exported ONNX graph's output shapes (e.g. with Netron).
constexpr int kNumQueries = 300;

// Must match model.image_normalize for the exported checkpoint.
// "dinov3/vitt16-ltdetr-coco" uses LT-DETR's default mean=(0,0,0),
// std=(1,1,1), so only /255 scaling applies. Always cross-check against the
// printed value from your own export.
const od_common::ImageNormalize kNormalize = {{0.0f, 0.0f, 0.0f},
                                               {1.0f, 1.0f, 1.0f}};

// Must match model.classes (in internal-index order) for the exported
// checkpoint -- print(model.classes) in Python and paste its values here in
// order. The list below is the standard 80-class COCO ordering used by
// "dinov3/vitt16-ltdetr-coco".
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

}  // namespace

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "od_infer_onnxruntime");
  Ort::SessionOptions session_options;

  OrtCUDAProviderOptions cuda_provider_options{};
  cuda_provider_options.device_id = kDeviceId;
  session_options.AppendExecutionProvider_CUDA(cuda_provider_options);

  Ort::Session session(env, kModelPath, session_options);

  const auto preprocessed = od_common::preprocess_image(
      kImagePath, kModelWidth, kModelHeight, kNormalize);

  // OrtDeviceAllocator (not OrtArenaAllocator): this MemoryInfo only
  // describes where our own cudaMalloc'd pointers live, ONNX Runtime never
  // allocates through it.
  Ort::MemoryInfo cuda_memory_info("Cuda", OrtDeviceAllocator, kDeviceId,
                                    OrtMemTypeDefault);

  const int num_classes = static_cast<int>(kClassNames.size());

  // Allocate the input and both outputs directly on the GPU -- this is what
  // makes this a zero-copy pipeline instead of relying on ONNX Runtime's
  // implicit host<->device copies, or its own allocator, inside
  // session.Run().
  float* d_input = nullptr;
  const size_t input_count = preprocessed.data.size();
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), input_count * sizeof(float)),
             "cudaMalloc(d_input)");
  check_cuda(cudaMemcpy(d_input, preprocessed.data.data(), input_count * sizeof(float),
                         cudaMemcpyHostToDevice),
             "cudaMemcpy(d_input)");

  float* d_logits = nullptr;
  const size_t logits_count = static_cast<size_t>(kNumQueries) * num_classes;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_logits), logits_count * sizeof(float)),
             "cudaMalloc(d_logits)");

  float* d_boxes = nullptr;
  const size_t boxes_count = static_cast<size_t>(kNumQueries) * 4;
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_boxes), boxes_count * sizeof(float)),
             "cudaMalloc(d_boxes)");

  const std::array<int64_t, 4> input_shape = {1, 3, kModelHeight, kModelWidth};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      cuda_memory_info, d_input, input_count, input_shape.data(), input_shape.size());

  const std::array<int64_t, 3> logits_shape = {1, kNumQueries, num_classes};
  Ort::Value logits_tensor = Ort::Value::CreateTensor<float>(
      cuda_memory_info, d_logits, logits_count, logits_shape.data(), logits_shape.size());

  const std::array<int64_t, 3> boxes_shape = {1, kNumQueries, 4};
  Ort::Value boxes_tensor = Ort::Value::CreateTensor<float>(
      cuda_memory_info, d_boxes, boxes_count, boxes_shape.data(), boxes_shape.size());

  Ort::IoBinding io_binding(session);
  io_binding.BindInput("images", input_tensor);
  io_binding.BindOutput("logits", logits_tensor);
  io_binding.BindOutput("boxes", boxes_tensor);

  session.Run(Ort::RunOptions{nullptr}, io_binding);
  io_binding.SynchronizeOutputs();

  std::vector<float> logits_host(logits_count);
  std::vector<float> boxes_host(boxes_count);
  check_cuda(cudaMemcpy(logits_host.data(), d_logits, logits_count * sizeof(float),
                         cudaMemcpyDeviceToHost),
             "cudaMemcpy(logits_host)");
  check_cuda(cudaMemcpy(boxes_host.data(), d_boxes, boxes_count * sizeof(float),
                         cudaMemcpyDeviceToHost),
             "cudaMemcpy(boxes_host)");

  const auto detections =
      od_common::postprocess(logits_host.data(), boxes_host.data(), kNumQueries,
                              num_classes, kNumTopQueries, preprocessed.orig_w,
                              preprocessed.orig_h,
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
  cudaFree(d_logits);
  cudaFree(d_boxes);
  return 0;
}
