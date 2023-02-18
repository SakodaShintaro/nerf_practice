#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>

#include "../data.hpp"
#include "../nerf_function.hpp"
#include "../nerf_model.hpp"
#include "../timer.hpp"
#include "../utils.hpp"

void TestGetRays() {
  const std::string dataset_path = "../data/train/greek/";
  CameraIntrinsicParameter param = GetCameraIntrinsicParameter(dataset_path);
  const std::vector<std::string> pose_paths = Glob(dataset_path + "pose/");
  const std::vector<std::string> image_paths = Glob(dataset_path + "rgb/");
  assert(pose_paths.size() == image_paths.size());
  const int32_t N = pose_paths.size();
  std::vector<Data> dataset_raw(N);
  const int32_t i = 0;
  std::cout << pose_paths[i] << " " << image_paths[i] << std::endl;
  dataset_raw[i].pose = ParsePose(pose_paths[i]);
  dataset_raw[i].image = cv::imread(image_paths[i]);
  std::vector<RayData> curr_rays = GetRays(param, dataset_raw[i]);
  std::cout << curr_rays[0].o << std::endl;
  std::cout << curr_rays[0].d << std::endl;
}

int main() { TestGetRays(); }
