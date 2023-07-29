#include "camera_intrinsic_parameter.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>

CameraIntrinsicParameter GetCameraIntrinsicParameter(const std::string& dataset_path) {
  YAML::Node camera_info = YAML::LoadFile(dataset_path + "/camera_info.yaml");
  YAML::Node intrinsic_node = camera_info["K"];

  float f = intrinsic_node[0].as<float>();
  float cx = intrinsic_node[2].as<float>();
  float cy = intrinsic_node[5].as<float>();

  float img_width = camera_info["width"].as<float>();
  float img_height = camera_info["height"].as<float>();

  std::cout << "Original" << std::endl;
  std::cout << "focal_length: " << f << std::endl;
  std::cout << "image_center: "
            << "(" << cx << ", " << cy << ")" << std::endl;
  std::cout << "image size  : "
            << "(" << img_height << ", " << img_width << ")" << std::endl;

  // データセットでの画像サイズ
  constexpr int32_t width = 512;
  constexpr int32_t height = 512;

  CameraIntrinsicParameter result;
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
