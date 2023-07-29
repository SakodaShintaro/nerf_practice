#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

#include "../data.hpp"
#include "../nerf_function.hpp"
#include "../nerf_model.hpp"
#include "../timer.hpp"
#include "../utils.hpp"

cv::Mat ToCvImage(torch::Tensor tensor) {
  tensor = tensor.cpu();
  const int32_t height = tensor.size(0);
  const int32_t width = tensor.size(1);
  cv::Mat output_mat(cv::Size(width, height), CV_8UC3, tensor.data_ptr<uchar>());
  return output_mat.clone();
}

int main() {
  const std::string dataset_path = "/home/sakoda/data/converted/logiee/20230609_base_link_logiee/";
  const std::vector<std::string> pose_paths = Glob(dataset_path + "pose/");
  const std::vector<std::string> image_paths = Glob(dataset_path + "images/");
  assert(pose_paths.size() == image_paths.size());
  const int32_t N = pose_paths.size();
  std::vector<Data> dataset_raw(N);
  for (int32_t i = 0; i < N; i++) {
    std::cout << pose_paths[i] << " " << image_paths[i] << std::endl;
    dataset_raw[i].pose = ParsePose(pose_paths[i]);
    dataset_raw[i].image = cv::imread(image_paths[i]);
  }

  CameraIntrinsicParameter param = GetCameraIntrinsicParameter(dataset_path);
  constexpr int32_t kDownScale = 16;
  param.f /= kDownScale;
  param.cx /= kDownScale;
  param.cy /= kDownScale;
  param.width /= kDownScale;
  param.height /= kDownScale;

  NeRF nerf;
  torch::load(nerf, "./result_dir/train/nerf_model.pt");
  nerf->to(torch::kCUDA);
  nerf->eval();

  const std::string result_dir = "result_dir/";
  const std::string save_dir = result_dir + "result_images/";
  std::filesystem::create_directories(save_dir);

  constexpr int32_t kNum = 64;
  constexpr int32_t kBatchSize = 2048;

  for (int32_t ind = 0; ind < N; ind++) {
    const float a = -M_PI + (2 * M_PI) / kNum * ind;
    const float c = std::cos(a);
    const float s = std::sin(a);
    Data data;
    data.pose = dataset_raw[ind].pose;
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
    C_all = torch::clamp(C_all, 0.0f, 1.0f) * 255;
    C_all = C_all.to(torch::kByte);
    cv::Mat image = ToCvImage(C_all);
    std::stringstream ss;
    ss << save_dir;
    ss << std::setfill('0') << std::setw(8) << ind;
    ss << ".png";
    const std::string save_path = ss.str();
    cv::imwrite(save_path, image);
    std::cout << "save " << save_path << std::endl;
  }
}
