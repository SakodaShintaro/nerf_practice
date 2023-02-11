#ifndef CAMERA_INTRINSIC_PARAMETER_HPP
#define CAMERA_INTRINSIC_PARAMETER_HPP

#include <vector>

#include "pose.hpp"

struct View {
  float f;
  float cx;
  float cy;
  std::vector<Pose> pose;
  int32_t width;
  int32_t height;
};

View GetView(const std::string& dataset_path);

#endif
