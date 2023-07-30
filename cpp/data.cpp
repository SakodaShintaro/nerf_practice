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

void normalize(std::vector<Data>& data) {
  // normalize (x, y, z) to [-1, 1]
  float mean_x = 0.0f;
  float mean_y = 0.0f;
  float mean_z = 0.0f;
  const int64_t n = data.size();
  for (int64_t i = 0; i < n; i++) {
    mean_x += data[i].pose(0, 3);
    mean_y += data[i].pose(1, 3);
    mean_z += data[i].pose(2, 3);
  }
  mean_x /= n;
  mean_y /= n;
  mean_z /= n;
  float max_norm = 0.0f;
  for (int64_t i = 0; i < n; i++) {
    data[i].pose(0, 3) -= mean_x;
    data[i].pose(1, 3) -= mean_y;
    data[i].pose(2, 3) -= mean_z;
    max_norm = std::max(max_norm, data[i].pose.block(0, 3, 3, 1).norm());
  }
  for (int64_t i = 0; i < n; i++) {
    data[i].pose.block(0, 3, 3, 1) /= max_norm;
  }
}
