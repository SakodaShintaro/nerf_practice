#include "data.hpp"

#include <fstream>

Pose ParsePose(const std::string& path) {
  std::ifstream ifs(path);
  Pose pose;
  for (int32_t i = 0; i < 16; i++) {
    ifs >> pose[i];
  }
  return pose;
}
