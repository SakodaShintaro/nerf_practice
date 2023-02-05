#ifndef DATA_HPP
#define DATA_HPP

#include <array>
#include <opencv2/opencv.hpp>

using Pose = std::array<float, 16>;

struct Data {
  Pose pose;
  cv::Mat rgb;
};

Pose ParsePose(const std::string& path);

#endif
