#include <iostream>
#include <opencv2/opencv.hpp>

#include "data.hpp"
#include "nerf_function.hpp"
#include "neural_network.hpp"
#include "utils.hpp"

int main() {
  std::cout << "NeRF Practice" << std::endl;
  const std::string dataset_path = "../data/train/greek/";

  const View view = GetView(dataset_path);

  const std::vector<std::string> pose_paths = Glob(dataset_path + "pose/");
  const std::vector<std::string> rgb_paths = Glob(dataset_path + "rgb/");
  assert(pose_paths.size() == rgb_paths.size());
  const int32_t N = pose_paths.size();
  std::vector<Data> dataset_raw(N);
  for (int32_t i = 0; i < N; i++) {
    std::cout << pose_paths[i] << " " << rgb_paths[i] << std::endl;
    dataset_raw[i].pose = ParsePose(pose_paths[i]);
    dataset_raw[i].rgb = cv::imread(rgb_paths[i]);
  }
}
