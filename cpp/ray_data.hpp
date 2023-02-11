#ifndef RAY_DATA_HPP
#define RAY_DATA_HPP

#include <opencv2/opencv.hpp>

#include "position.hpp"

struct RayData {
  Position o;     // カメラの位置（焦点位置）
  Position d;     // カメラの視線ベクトル
  cv::Vec3b bgr;  // ピクセルの色
};

#endif
