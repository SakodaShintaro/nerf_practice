﻿#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  std::cout << "NeRF Practice" << std::endl;
  cv::Mat img = cv::imread("/root/nerf_practice/data/test/greek/rgb/00000.png");
  if (img.empty()) {
    return 1;
  }
  cv::imwrite("qwe.png", img);
}
