#include <iostream>
#include <opencv2/opencv.hpp>

#include "nerf_function.hpp"
#include "neural_network.hpp"

int main() {
  std::cout << "NeRF Practice" << std::endl;
  cv::Mat img = cv::imread("/root/nerf_practice/data/test/greek/rgb/00000.png");
  if (img.empty()) {
    return 1;
  }
  cv::imwrite("qwe.png", img);
}
