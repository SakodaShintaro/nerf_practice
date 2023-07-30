#ifndef BASIC_RADIANCE_FIELD_HPP
#define BASIC_RADIANCE_FIELD_HPP

#include <torch/torch.h>

class BasicRadianceFieldImpl : public torch::nn::Module {
 public:
  BasicRadianceFieldImpl();

  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor d);

 private:
  torch::Tensor gamma(torch::Tensor p, int32_t L);

  const int32_t L_x = 10;
  const int32_t L_d = 4;
  torch::nn::Linear layer0_{nullptr};
  torch::nn::Linear layer1_{nullptr};
  torch::nn::Linear layer2_{nullptr};
  torch::nn::Linear layer3_{nullptr};
  torch::nn::Linear layer4_{nullptr};
  torch::nn::Linear layer5_{nullptr};
  torch::nn::Linear layer6_{nullptr};
  torch::nn::Linear layer7_{nullptr};
  torch::nn::Linear sigma_{nullptr};
  torch::nn::Linear layer8_{nullptr};
  torch::nn::Linear layer9_{nullptr};
  torch::nn::Linear layer10_{nullptr};
  torch::nn::Linear layer11_{nullptr};
  torch::nn::Linear layer12_{nullptr};
  torch::nn::Linear rgb_{nullptr};
};

TORCH_MODULE(BasicRadianceField);

#endif
