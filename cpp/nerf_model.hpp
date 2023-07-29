#ifndef NERF_MODEL_HPP
#define NERF_MODEL_HPP

#include <opencv2/opencv.hpp>

#include "camera_intrinsic_parameter.hpp"
#include "radiance_field.hpp"

class NeRFImpl : public torch::nn::Module {
 public:
  NeRFImpl(float _t_n = 0.0f, float _t_f = 2.5f, int32_t _L_x = 10, int32_t _L_d = 4,
           cv::Vec3b _c_bg = {255, 255, 255});
  torch::Device device();
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor o, torch::Tensor d);

 private:
  static constexpr int32_t kNumCoarse = 64;
  static constexpr int32_t kNumFine = 128;
  const float t_near_;
  const float t_far_;
  const cv::Vec3b back_ground_color_;
  RadianceField rf_coarse_{nullptr};
  RadianceField rf_fine_{nullptr};
};

TORCH_MODULE(NeRF);

#endif
