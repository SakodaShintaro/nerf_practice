#include <iostream>
#include <opencv2/opencv.hpp>

#include "nerf_function.hpp"
#include "neural_network.hpp"
#include "utils.hpp"

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

  std::cout << "Original" << std::endl;
  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << img_height << ", " << img_width << ")" << std::endl;

  // データセットでの画像サイズ
  constexpr int32_t width = 512;
  constexpr int32_t height = 512;

  f *= height / img_height;
  cx *= width / img_width;
  cy *= height / img_height;
  std::cout << "Resized" << std::endl;
  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << height << ", " << width << ")" << std::endl;

  const std::vector<std::string> pose_paths = Glob(dataset_path + "pose/");
  const std::vector<std::string> rgb_paths = Glob(dataset_path + "rgb/");
  assert(pose_paths.size() == rgb_paths.size());
  const int32_t N = pose_paths.size();
  for (int32_t i = 0; i < N; i++) {
    std::cout << pose_paths[i] << " " << rgb_paths[i] << std::endl;
  }
}
