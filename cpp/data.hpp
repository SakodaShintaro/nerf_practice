#ifndef DATA_HPP
#define DATA_HPP

#include <array>
#include <opencv2/opencv.hpp>

#include "pose.hpp"

struct Data {
  Pose pose;
  cv::Mat image;
};

Pose ParsePose(const std::string& path);

#endif
