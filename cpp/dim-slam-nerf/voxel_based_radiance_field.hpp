// The original python implementation is https://github.com/HKUST-3DV/DIM-SLAM/blob/main/src/nerf.py

#ifndef DIM_SLAM_NERF__VOXEL_BASED_RADIANCE_FIELD_HPP_
#define DIM_SLAM_NERF__VOXEL_BASED_RADIANCE_FIELD_HPP_

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

class VoxelBasedRadianceFieldImpl : public torch::nn::Module {
 public:
  VoxelBasedRadianceFieldImpl();
  torch::Tensor normalize_3d_coordinate(torch::Tensor p, double grid_len, std::vector<int64_t> grid_shape);
  std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor x, torch::Tensor d);

 private:
  std::vector<torch::Tensor> grids_feat_;
  std::vector<std::vector<int64_t>> grids_shape_;
  std::vector<double> grids_lens_;
  torch::Tensor bound_;
  BasicMLP alpha_decoder_ = nullptr;
  BasicMLP color_decoder_ = nullptr;
};
TORCH_MODULE(VoxelBasedRadianceField);

#endif
