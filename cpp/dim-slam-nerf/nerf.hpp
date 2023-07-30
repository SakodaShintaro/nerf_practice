// The original python implementation is https://github.com/HKUST-3DV/DIM-SLAM/blob/main/src/nerf.py

#ifndef DIM_SLAM_NERF__NERF_HPP_
#define DIM_SLAM_NERF__NERF_HPP_

#include <torch/torch.h>

class BasicMLPImpl : public torch::nn::Module {
 public:
  BasicMLPImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim, int64_t n_blocks, int64_t skip);
  torch::Tensor forward(torch::Tensor input);

 private:
  torch::nn::ModuleList layers_;
  int64_t skip_;
};
TORCH_MODULE(BasicMLP);

class NeRFImpl : public torch::nn::Module {
 public:
  NeRFImpl();
  torch::Tensor normalize_3d_coordinate(torch::Tensor p, double grid_len, std::vector<int64_t> grid_shape);
  torch::Tensor forward(torch::Tensor input);

 private:
  std::vector<torch::Tensor> grids_feat_;
  std::vector<std::vector<int64_t>> grids_shape_;
  std::vector<double> grids_lens_;
  torch::Tensor bound_;
  std::vector<torch::Tensor> grids_xyz_;
  BasicMLP alpha_decoder_ = nullptr;
  BasicMLP color_decoder_ = nullptr;
};
TORCH_MODULE(NeRF);

#endif
