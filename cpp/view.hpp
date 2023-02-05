#ifndef VIEW_HPP
#define VIEW_HPP

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

#endif