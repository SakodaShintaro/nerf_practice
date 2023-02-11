#include "data.hpp"

#include <fstream>

Pose ParsePose(const std::string& path) {
  std::ifstream ifs(path);
  Pose pose;
  for (int32_t i = 0; i < 4; i++) {
    for (int32_t j = 0; j < 4; j++) {
      ifs >> pose(i, j);
    }
  }
  return pose;
}
