#include "nerf_model.hpp"

#include "nerf_function.hpp"

NeRFImpl::NeRFImpl(float _t_n, float _t_f, int32_t _L_x, int32_t _L_d, cv::Vec3b _c_bg)
    : t_n(_t_n), t_f(_t_f), c_bg(_c_bg) {
  using namespace torch::nn;
  rf_c = register_module("rf_c", RadianceField(_L_x, _L_d));
  rf_f = register_module("rf_f", RadianceField(_L_x, _L_d));
}

torch::Device NeRFImpl::device() { return rf_c->parameters().front().device(); }

std::pair<torch::Tensor, torch::Tensor> NeRFImpl::forward(torch::Tensor o, torch::Tensor d) {
  const torch::Device dev = device();
  o = o.to(dev);
  d = d.to(dev);

  const int32_t batch_size = o.size(0);

  torch::Tensor partitions = SplitRay(t_n, t_f, N_c, batch_size);
  partitions = partitions.to(dev);

  std::vector<float> bg_vec(c_bg.val, c_bg.val + 3);
  torch::Tensor bg = torch::tensor(bg_vec, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  bg = bg.view({1, 3});
  bg /= 255;

  torch::Tensor t_c = SampleCoarse(partitions).to(dev);
  t_c = t_c.to(dev);

  auto [rgb_c, w_c] = _rgb_and_weight(rf_c, o, d, t_c, N_c);
  torch::Tensor C_c = torch::sum(w_c.unsqueeze(-1) * rgb_c, 1);
  C_c += (1.0f - torch::sum(w_c, 1, true)) * bg;

  w_c = w_c.clone();
  torch::Tensor t_f = SampleFine(partitions, w_c, t_c, N_f).to(dev);

  auto [rgb_f, w_f] = _rgb_and_weight(rf_f, o, d, t_f, N_f + N_c);
  torch::Tensor C_f = torch::sum(w_f.unsqueeze(-1) * rgb_f, 1);
  C_f += (1.0f - torch::sum(w_f, 1, true)) * bg;

  return std::make_pair(C_c, C_f);
}
