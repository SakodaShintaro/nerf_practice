#ifndef NERF_MODEL_HPP
#define NERF_MODEL_HPP

#include "camera_intrinsic_parameter.hpp"
#include "radiance_field.hpp"

class NeRF : public torch::nn::Module {
 public:
  NeRF();
  torch::Device device();
  std::pair<torch::Tensor, torch::Tensor> forward(const CameraIntrinsicParameter& param);

 private:
  static constexpr int32_t N_c = 64;
  static constexpr int32_t N_f = 128;
  static constexpr int32_t N_SAMPLES = 2048;
  float t_n;
  float t_f;
  cv::Vec3b c_bg;  // 背景の色
};

#endif
