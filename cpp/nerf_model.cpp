#include "nerf_model.hpp"

#include "nerf_function.hpp"

NeRFImpl::NeRFImpl(float _t_n, float _t_f, int32_t _L_x, int32_t _L_d, cv::Vec3b _c_bg)
    : t_near_(_t_n), t_far_(_t_f), back_ground_color_(_c_bg) {
  using namespace torch::nn;
  rf_coarse_ = register_module("rf_coarse_", RadianceField(_L_x, _L_d));
  rf_fine_ = register_module("rf_fine_", RadianceField(_L_x, _L_d));
}

torch::Device NeRFImpl::device() { return rf_coarse_->parameters().front().device(); }

std::pair<torch::Tensor, torch::Tensor> NeRFImpl::forward(torch::Tensor o, torch::Tensor d) {
  const torch::Device dev = device();
  o = o.to(dev);
  d = d.to(dev);

  const int32_t batch_size = o.size(0);

  torch::Tensor partitions = SplitRay(t_near_, t_far_, kNumCoarse, batch_size);
  partitions = partitions.to(dev);

  std::vector<float> bg_vec(back_ground_color_.val, back_ground_color_.val + 3);
  torch::Tensor bg = torch::tensor(bg_vec, torch::TensorOptions().device(dev).dtype(torch::kFloat32));
  bg = bg.view({1, 3});
  bg /= 255;

  torch::Tensor t_c = SampleCoarse(partitions).to(dev);
  t_c = t_c.to(dev);

  auto [rgb_c, w_c] = _rgb_and_weight(rf_coarse_, o, d, t_c, kNumCoarse);
  torch::Tensor C_c = torch::sum(w_c.unsqueeze(-1) * rgb_c, 1);
  C_c += (1.0f - torch::sum(w_c, 1, true)) * bg;

  w_c = w_c.clone();
  torch::Tensor t_f = SampleFine(partitions, w_c, t_c, kNumFine).to(dev);

  auto [rgb_f, w_f] = _rgb_and_weight(rf_fine_, o, d, t_f, kNumFine + kNumCoarse);
  torch::Tensor C_f = torch::sum(w_f.unsqueeze(-1) * rgb_f, 1);
  C_f += (1.0f - torch::sum(w_f, 1, true)) * bg;

  return std::make_pair(C_c, C_f);
}
