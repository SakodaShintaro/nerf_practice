#ifndef CAMERA_INTRINSIC_PARAMETER_HPP
#define CAMERA_INTRINSIC_PARAMETER_HPP

#include <vector>

#include "pose.hpp"

struct CameraIntrinsicParameter {
  float f;
  float cx;
  float cy;
  int32_t width;
  int32_t height;
};

CameraIntrinsicParameter GetCameraIntrinsicParameter(const std::string& dataset_path);

#endif
