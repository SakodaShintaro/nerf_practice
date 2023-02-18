#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

#include "../data.hpp"
#include "../nerf_function.hpp"
#include "../nerf_model.hpp"
#include "../timer.hpp"
#include "../utils.hpp"

int main() {
  const std::string dataset_path = "../data/train/greek/";

  CameraIntrinsicParameter param = GetCameraIntrinsicParameter(dataset_path);
  param.f /= 2;
  param.cx /= 2;
  param.cy /= 2;
  param.width /= 2;
  param.height /= 2;

  NeRF nerf;
  torch::load(nerf, "result_dir/train/nerf_model.pt");
  nerf->to(torch::kCUDA);

  const std::string result_dir = "result_dir/";
  const std::string save_dir = result_dir + "result_images/";
  std::filesystem::create_directories(save_dir);

  constexpr int32_t kNum = 64;

  Pose base_pose;

  for (int32_t ind = 0; ind < kNum; ind++) {
    const float a = -M_PI + (2 * M_PI) / kNum;
    const float c = std::cos(a);
    const float s = std::sin(a);
    Data data;
    data.pose = Eigen::Matrix4f::Identity();
    data.pose(0, 0) = c;
    data.pose(0, 2) = -s;
    data.pose(2, 0) = s;
    data.pose(2, 2) = c;
    data.pose = data.pose * base_pose;
    std::vector<RayData> ray_data = GetRays(param, data);

  }
}
