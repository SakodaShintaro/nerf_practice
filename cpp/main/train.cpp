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

  const CameraIntrinsicParameter param = GetCameraIntrinsicParameter(dataset_path);

  const std::vector<std::string> pose_paths = Glob(dataset_path + "pose/");
  const std::vector<std::string> image_paths = Glob(dataset_path + "rgb/");
  assert(pose_paths.size() == image_paths.size());
  const int32_t N = pose_paths.size();
  std::vector<Data> dataset_raw(N);
  std::vector<RayData> ray_data;
  for (int32_t i = 0; i < N; i++) {
    std::cout << pose_paths[i] << " " << image_paths[i] << std::endl;
    dataset_raw[i].pose = ParsePose(pose_paths[i]);
    dataset_raw[i].image = cv::imread(image_paths[i]);
    std::vector<RayData> curr_rays = GetRays(param, dataset_raw[i]);
    ray_data.insert(ray_data.end(), curr_rays.begin(), curr_rays.end());
  }

  constexpr int32_t kEpoch = 1;
  constexpr int32_t kBatchSize = 2048;
  constexpr int32_t kMaxStep = 1000;
  constexpr int32_t kPrintInterval = 100;

  NeRF nerf;
  nerf->to(torch::kCUDA);
  torch::optim::AdamOptions options;
  options.lr(3e-4);
  options.betas({0.9, 0.999});
  options.eps(1e-7);
  torch::optim::Adam optimizer(nerf->parameters(), options);

  const int32_t n_sample = ray_data.size();
  std::cout << "n_sample = " << n_sample << std::endl;

  const std::string result_dir = "result_dir/";
  const std::string save_dir = result_dir + "train/";
  std::filesystem::create_directories(save_dir);
  std::ofstream ofs(save_dir + "train_loss.tsv");
  const std::string header_str = "time\tepoch\tstep\tepoch_rate\tloss\n";
  ofs << header_str;
  std::cout << header_str;

  int32_t step = 0;
  Timer timer;
  timer.Start();

  std::mt19937_64 engine(std::random_device{}());
  for (int32_t e = 1; e <= kEpoch; e++) {
    std::shuffle(ray_data.begin(), ray_data.end(), engine);
    float sum_loss = 0.0f;
    int64_t sample_num = 0;

    for (int32_t i = 0; i < n_sample; i += kBatchSize) {
      step++;

      // ミニバッチを作成
      const std::vector<RayData> curr_data(ray_data.begin() + i, ray_data.begin() + std::min(i + kBatchSize, n_sample));
      auto [o, d, C] = RayData2Tensor(curr_data);
      C = C.to(nerf->device());
      C /= 255;

      // 推論
      auto [C_c, C_f] = nerf->forward(o, d);
      torch::Tensor loss = torch::nn::functional::mse_loss(C_c, C) + torch::nn::functional::mse_loss(C_f, C);
      sum_loss += loss.item<float>() * o.size(0);
      sample_num += o.size(0);

      if ((i / kBatchSize + 1) % kPrintInterval == 0) {
        sum_loss /= sample_num;
        const int64_t elapsed_sec = timer.ElapsedSeconds();
        const int64_t elapsed_min = elapsed_sec / 60 % 60;
        const int64_t elapsed_hou = elapsed_sec / 3600;
        std::stringstream ss;
        ss << std::setfill('0') << std::fixed;
        ss << std::setw(2) << elapsed_hou << ":";
        ss << std::setw(2) << elapsed_min << ":";
        ss << std::setw(2) << elapsed_sec % 60 << "\t";
        const float epoch_rate = 100.0f * (i + kBatchSize) / n_sample;
        ss << e << "\t";
        ss << step << "\t";
        ss << epoch_rate << "\t";
        ss << sum_loss << "\t";
        ofs << ss.str() << std::endl;
        std::cout << ss.str() << std::endl;
        sum_loss = 0;
        sample_num = 0;
      }

      optimizer.zero_grad();
      loss.backward();
      optimizer.step();
      torch::save(nerf, save_dir + "nerf_model.pt");
      if (step >= kMaxStep) {
        break;
      }
    }

    torch::save(nerf, save_dir + "nerf_model.pt");
    torch::save(optimizer, save_dir + "optimizer.pt");
    if (step >= kMaxStep) {
      break;
    }
  }
}
