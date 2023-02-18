#ifndef RAY_DATA_HPP
#define RAY_DATA_HPP

#include <torch/torch.h>

#include <opencv2/opencv.hpp>

#include "position.hpp"

struct RayData {
  Position o;     // カメラの位置（焦点位置）
  Position d;     // カメラの視線ベクトル
  cv::Vec3b bgr;  // ピクセルの色
};

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RayData2Tensor(const std::vector<RayData>& data);

#endif
