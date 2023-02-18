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
  torch::load(nerf, "./result_dir/train/nerf_model.pt");
  nerf->to(torch::kCUDA);
  nerf->eval();

  const std::string result_dir = "result_dir/";
  const std::string save_dir = result_dir + "result_images/";
  std::filesystem::create_directories(save_dir);

  constexpr int32_t kNum = 64;
  constexpr int32_t kBatchSize = 2048;

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
    data.image = cv::Mat::zeros(param.height, param.width, CV_32F);
    std::vector<RayData> ray_data = GetRays(param, data);
    const int32_t n_sample = ray_data.size();

    // 推論
    torch::NoGradGuard no_grad_guard;
    torch::Tensor C_all;
    for (int32_t i = 0; i < n_sample; i += kBatchSize) {
      const std::vector<RayData> curr_data(ray_data.begin() + i, ray_data.begin() + std::min(i + kBatchSize, n_sample));
      auto [o, d, C] = RayData2Tensor(curr_data);
      auto [C_c, C_f] = nerf->forward(o, d);

      C_all = (i == 0 ? C_f : torch::concatenate({C_all, C_f}, 0));
    }
    C_all = C_all.reshape({param.height, param.width, 3});
    C_all = torch::clamp(C_all, 0.0f, 1.0f);

    std::cout << C_all.sizes() << std::endl;
  }
}
