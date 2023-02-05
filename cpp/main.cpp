#include <iostream>
#include <opencv2/opencv.hpp>

#include "nerf_function.hpp"
#include "neural_network.hpp"

int main() {
  std::cout << "NeRF Practice" << std::endl;
  const std::string dataset_path = "../data/train/greek/";

  std::ifstream ifs_intrinsics(dataset_path + "intrinsics.txt");

  // 1行目
  float f, cx, cy, _;
  ifs_intrinsics >> f >> cx >> cy >> _;

  // 2行目
  float origin_x, origin_y, origin_z;
  ifs_intrinsics >> origin_x >> origin_y >> origin_z;

  // 3行目
  float near_plane;
  ifs_intrinsics >> near_plane >> _;

  // 4行目
  float img_height, img_width;
  ifs_intrinsics >> img_height >> img_width;

  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << img_height << ", " << img_width << ")" << std::endl;
}
