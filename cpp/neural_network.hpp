#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "nerf_function.hpp"

torch::Tensor gamma(torch::Tensor p, int32_t L);

class RadianceField : public torch::nn::Module {
 public:
  RadianceField(int32_t L_x = 10, int32_t L_d = 4);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor d);

 private:
  torch::nn::Linear layer0_;
  torch::nn::Linear layer1_;
  torch::nn::Linear layer2_;
  torch::nn::Linear layer3_;
  torch::nn::Linear layer4_;
  torch::nn::Linear layer5_;
  torch::nn::Linear layer6_;
  torch::nn::Linear layer7_;
  torch::nn::Linear sigma_;
  torch::nn::Linear layer8_;
  torch::nn::Linear layer9_;
  torch::nn::Linear layer10_;
  torch::nn::Linear layer11_;
  torch::nn::Linear layer12_;
  torch::nn::Linear rgb_;
};

#endif
