//
// Copyright (c) Lightly AG and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Shared preprocessing/postprocessing for the LT-DETR object detection C++
// recipes (ONNX Runtime + TensorRT). Ports the same math as
// lightly_train._pre_post_processing.object_detection.ObjectDetectionPreprocessor
// and ObjectDetectionPostprocessor.decode()/postprocess() so both recipes stay
// numerically consistent with each other and with `model.predict(...)`.
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace od_common {

struct ImageNormalize {
  std::array<float, 3> mean;
  std::array<float, 3> std;
};

struct PreprocessResult {
  std::vector<float> data;  // CHW, fp32, length 3 * model_h * model_w
  int orig_w;
  int orig_h;
};

// Loads `image_path`, resizes to the model's fixed (model_w, model_h) input
// size (exact resize, NOT aspect-preserving, matching
// ObjectDetectionPreprocessor.preprocess_image/preprocess_batch), converts
// BGR -> RGB, scales to [0,1], applies per-channel (x - mean) / std, and
// returns the result as a flat CHW float32 buffer plus the original image
// size (needed later to rescale predicted boxes back to the original image).
inline PreprocessResult preprocess_image(const std::string& image_path, int model_w,
                                          int model_h,
                                          const ImageNormalize& normalize) {
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    throw std::runtime_error("Failed to read image: " + image_path);
  }

  const int orig_w = image.cols;
  const int orig_h = image.rows;

  cv::Mat resized;
  // cv::Size is (width, height) -- do not swap with the model's (H, W) order.
  cv::resize(image, resized, cv::Size(model_w, model_h), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  cv::Mat float_image;
  resized.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

  PreprocessResult result;
  result.orig_w = orig_w;
  result.orig_h = orig_h;
  result.data.resize(static_cast<size_t>(3) * model_h * model_w);

  const size_t plane = static_cast<size_t>(model_h) * model_w;
  for (int h = 0; h < model_h; ++h) {
    const auto* row = float_image.ptr<cv::Vec3f>(h);
    for (int w = 0; w < model_w; ++w) {
      const cv::Vec3f& px = row[w];
      for (int c = 0; c < 3; ++c) {
        result.data[c * plane + static_cast<size_t>(h) * model_w + w] =
            (px[c] - normalize.mean[c]) / normalize.std[c];
      }
    }
  }
  return result;
}

struct Detection {
  int label;  // internal class index (0..num_classes-1), see note in postprocess()
  float x1, y1, x2, y2;  // xyxy, ORIGINAL image pixel coordinates
  float score;
};

// Ports ObjectDetectionPostprocessor.decode() + postprocess() verbatim
// (src/lightly_train/_pre_post_processing/object_detection.py:135-171):
//
//   scores = sigmoid(logits)
//   (scores, index) = topk(flatten(scores, num_queries * num_classes), num_top_queries)
//   label = index % num_classes; query_index = index / num_classes
//   box = cxcywh_to_xyxy(boxes[query_index]) * [orig_w, orig_h, orig_w, orig_h]
//   keep where score > threshold
//
// `logits` is (num_queries, num_classes) and `boxes` is (num_queries, 4)
// normalized cxcywh in [0,1], both for a single image (batch dim already
// stripped by the caller). Because `boxes` is normalized (not pixel
// coordinates in the resized input image), scaling directly by the
// *original* image size is correct -- there is no intermediate
// resized-image pixel space to rescale from.
//
// `label` is the model's *internal* contiguous class index. Indexing a
// class-name array ordered the same way as `list(model.classes.values())`
// with this internal label directly reproduces `.predict()`'s human-readable
// output without needing the `internal_class_to_class` remap table, which
// only matters if you need the arbitrary public integer class id itself.
inline std::vector<Detection> postprocess(const float* logits, const float* boxes,
                                           int num_queries, int num_classes,
                                           int num_top_queries, int orig_w,
                                           int orig_h, float threshold) {
  const int total = num_queries * num_classes;

  std::vector<float> scores(static_cast<size_t>(total));
  for (int i = 0; i < total; ++i) {
    scores[i] = 1.f / (1.f + std::exp(-logits[i]));
  }

  std::vector<int> idx(static_cast<size_t>(total));
  std::iota(idx.begin(), idx.end(), 0);
  const int k = std::min(num_top_queries, total);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                     [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<Detection> out;
  out.reserve(static_cast<size_t>(k));
  for (int i = 0; i < k; ++i) {
    const float score = scores[idx[i]];
    if (score <= threshold) continue;

    const int label = idx[i] % num_classes;
    const int query_index = idx[i] / num_classes;

    const float cx = boxes[query_index * 4 + 0];
    const float cy = boxes[query_index * 4 + 1];
    const float w = boxes[query_index * 4 + 2];
    const float h = boxes[query_index * 4 + 3];

    out.push_back({label, (cx - w / 2.f) * orig_w, (cy - h / 2.f) * orig_h,
                   (cx + w / 2.f) * orig_w, (cy + h / 2.f) * orig_h, score});
  }
  return out;
}

// Draws xyxy boxes + "<name> <score>" labels onto `image` in place, mirroring
// the notebooks' `visualize_detections()` helper.
inline void draw_detections(cv::Mat& image, const std::vector<Detection>& detections,
                             const std::vector<std::string>& class_names) {
  for (const auto& det : detections) {
    const cv::Point top_left(static_cast<int>(det.x1), static_cast<int>(det.y1));
    const cv::Point bottom_right(static_cast<int>(det.x2), static_cast<int>(det.y2));
    cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2);

    const std::string name = (det.label >= 0 &&
                               static_cast<size_t>(det.label) < class_names.size())
                                  ? class_names[static_cast<size_t>(det.label)]
                                  : std::to_string(det.label);
    char score_buf[16];
    std::snprintf(score_buf, sizeof(score_buf), "%.2f", det.score);
    const std::string label_text = name + " " + score_buf;

    const cv::Point label_origin(top_left.x,
                                  std::max(top_left.y - 5, 10));
    cv::putText(image, label_text, label_origin, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
  }
}

}  // namespace od_common
