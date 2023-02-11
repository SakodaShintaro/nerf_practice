#include "view.hpp"
#include <fstream>
#include <iostream>

View GetView(const std::string& dataset_path) {
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

  std::cout << "Original" << std::endl;
  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << img_height << ", " << img_width << ")" << std::endl;

  // データセットでの画像サイズ
  constexpr int32_t width = 512;
  constexpr int32_t height = 512;

  View result;
  result.f = f * height / img_height;
  result.cx = cx * width / img_width;
  result.cy = cy * height / img_height;
  result.width = width;
  result.height = height;
  std::cout << "Resized" << std::endl;
  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << height << ", " << width << ")" << std::endl;
  return result;
}