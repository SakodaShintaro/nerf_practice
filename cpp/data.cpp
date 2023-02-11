#include "data.hpp"

#include <fstream>

Pose ParsePose(const std::string& path) {
  std::ifstream ifs(path);
  Pose pose;
  for (int32_t i = 0; i < 16; i++) {
    float v;
    ifs >> v;
    pose << v;
  }
  return pose;
}
