#ifndef NERF_MODEL_HPP
#define NERF_MODEL_HPP

#include <torch/torch.h>

#include "camera_intrinsic_parameter.hpp"

class NeRF : public torch::nn::Module {
 public:
  NeRF();
  torch::Device device();
  std::pair<torch::Tensor, torch::Tensor> forward(const CameraIntrinsicParameter& param);

 private:
  static constexpr int32_t N_c = 64;
  static constexpr int32_t N_f = 128;
  static constexpr int32_t N_SAMPLES = 2048;
};

#endif
