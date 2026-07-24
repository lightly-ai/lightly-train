//
// Copyright (c) Lightly AG and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Recipe: run LT-DETR object detection inference with ONNX Runtime's CUDA
// execution provider, allocating the model input directly on the GPU
// (zero-copy) via Ort::IoBinding.
//
// This is a single-purpose example, not a general-purpose CLI tool: edit the
// constants below to point at your own exported model/image, then rebuild.
//
// Export the model first, on a machine with lightly-train installed:
//
//   import lightly_train
//   model = lightly_train.load_model("ltdetrv2-s-coco")
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

// Must match model.image_normalize for the exported checkpoint. "ltdetrv2-s-coco"
// uses ImageNet statistics; other checkpoints may use LT-DETR's generic
// default of mean=(0,0,0), std=(1,1,1) (i.e. only the /255 scaling applies) --
// always cross-check against the printed value from your own export.
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

}  // namespace

int main() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "od_infer_onnxruntime");
  Ort::SessionOptions session_options;

  OrtCUDAProviderOptions cuda_provider_options{};
  cuda_provider_options.device_id = kDeviceId;
  session_options.AppendExecutionProvider_CUDA(cuda_provider_options);

  Ort::Session session(env, kModelPath, session_options);

  const auto pre = od_common::preprocess_image(kImagePath, kModelWidth, kModelHeight,
                                                kNormalize);

  Ort::MemoryInfo cuda_memory_info("Cuda", OrtArenaAllocator, kDeviceId,
                                    OrtMemTypeDefault);

  // Allocate the input directly on the GPU and copy the preprocessed image
  // there -- this, together with binding the outputs to CUDA memory below,
  // is what makes this a zero-copy pipeline instead of relying on ONNX
  // Runtime's implicit host<->device copies inside session.Run().
  float* d_input = nullptr;
  const size_t input_count = pre.data.size();
  check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_input), input_count * sizeof(float)),
             "cudaMalloc(d_input)");
  check_cuda(cudaMemcpy(d_input, pre.data.data(), input_count * sizeof(float),
                         cudaMemcpyHostToDevice),
             "cudaMemcpy(d_input)");

  const std::array<int64_t, 4> input_shape = {1, 3, kModelHeight, kModelWidth};
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      cuda_memory_info, d_input, input_count, input_shape.data(), input_shape.size());

  Ort::IoBinding io_binding(session);
  io_binding.BindInput("images", input_tensor);
  // The memory-info overload of BindOutput tells ONNX Runtime to allocate
  // the outputs on the GPU itself, rather than always materializing them on
  // host as a plain session.Run() call would.
  io_binding.BindOutput("logits", cuda_memory_info);
  io_binding.BindOutput("boxes", cuda_memory_info);

  session.Run(Ort::RunOptions{nullptr}, io_binding);
  io_binding.SynchronizeOutputs();

  const std::vector<std::string> output_names = io_binding.GetOutputNames();
  std::vector<Ort::Value> outputs = io_binding.GetOutputValues();

  const Ort::Value* logits_value = nullptr;
  const Ort::Value* boxes_value = nullptr;
  for (size_t i = 0; i < output_names.size(); ++i) {
    if (output_names[i] == "logits") {
      logits_value = &outputs[i];
    } else if (output_names[i] == "boxes") {
      boxes_value = &outputs[i];
    }
  }
  if (logits_value == nullptr || boxes_value == nullptr) {
    throw std::runtime_error("Model did not produce the expected 'logits'/'boxes' outputs.");
  }

  const auto logits_shape = logits_value->GetTensorTypeAndShapeInfo().GetShape();
  const int num_queries = static_cast<int>(logits_shape[1]);
  const int num_classes = static_cast<int>(logits_shape[2]);

  std::vector<float> logits_host(static_cast<size_t>(num_queries) * num_classes);
  std::vector<float> boxes_host(static_cast<size_t>(num_queries) * 4);
  check_cuda(cudaMemcpy(logits_host.data(), logits_value->GetTensorData<float>(),
                         logits_host.size() * sizeof(float), cudaMemcpyDeviceToHost),
             "cudaMemcpy(logits_host)");
  check_cuda(cudaMemcpy(boxes_host.data(), boxes_value->GetTensorData<float>(),
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
  return 0;
}
