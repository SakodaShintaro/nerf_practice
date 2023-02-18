#include "radiance_field.hpp"

RadianceFieldImpl::RadianceFieldImpl(const int32_t _L_x, const int32_t _L_d) : L_x(_L_x), L_d(_L_d) {
  using namespace torch::nn;
  layer0_ = register_module("layer0_", Linear(6 * L_x, 256));
  layer1_ = register_module("layer1_", Linear(256, 256));
  layer2_ = register_module("layer2_", Linear(256, 256));
  layer3_ = register_module("layer3_", Linear(256, 256));
  layer4_ = register_module("layer4_", Linear(256, 256));
  layer5_ = register_module("layer5_", Linear(256 + 6 * L_x, 256));
  layer6_ = register_module("layer6_", Linear(256, 256));
  layer7_ = register_module("layer7_", Linear(256, 256));
  sigma_ = register_module("sigma_", Linear(256, 1));
  layer8_ = register_module("layer8_", Linear(256, 256));
  layer9_ = register_module("layer9_", Linear(256 + 6 * L_d, 128));
  layer10_ = register_module("layer10_", Linear(128, 128));
  layer11_ = register_module("layer11_", Linear(128, 128));
  layer12_ = register_module("layer12_", Linear(128, 128));
  rgb_ = register_module("rgb_", Linear(128, 3));
}

std::pair<torch::Tensor, torch::Tensor> RadianceFieldImpl::forward(torch::Tensor x, torch::Tensor d) {
  torch::Tensor e_x = gamma(x, L_x);
  torch::Tensor e_d = gamma(x, L_d);

  torch::Tensor h;
  h = torch::relu(layer0_(e_x));
  h = torch::relu(layer1_(h));
  h = torch::relu(layer2_(h));
  h = torch::relu(layer3_(h));
  h = torch::relu(layer4_(h));
  h = torch::cat({h, e_x}, 1);
  h = torch::relu(layer5_(h));
  h = torch::relu(layer6_(h));
  h = torch::relu(layer7_(h));
  torch::Tensor sigma = torch::relu(sigma_(h));
  h = layer8_(h);
  h = torch::cat({h, e_d}, 1);
  h = torch::relu(layer9_(h));
  h = torch::relu(layer10_(h));
  h = torch::relu(layer11_(h));
  h = torch::relu(layer12_(h));
  torch::Tensor rgb = torch::sigmoid(rgb_(h));
  return std::make_pair(rgb, sigma);
}

torch::Tensor RadianceFieldImpl::gamma(torch::Tensor p, int32_t L) {
  p = torch::tanh(p);
  const int32_t batch_size = p.size(0);
  torch::Tensor i = torch::arange(0, L, torch::TensorOptions().device(p.device()).dtype(torch::kFloat32));
  i = i.view({1, 1, -1});
  p = p.view({batch_size, -1, 1});
  torch::Tensor a = torch::pow(2.0, i) * M_PI * p;
  torch::Tensor s = torch::sin(a);
  torch::Tensor c = torch::cos(a);
  torch::Tensor e = torch::cat({s, c}, 2).view({batch_size, -1});
  return e;
}
