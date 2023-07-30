#include "voxel_based_radiance_field.hpp"

using namespace torch::indexing;

BasicMLPImpl::BasicMLPImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim, int64_t n_blocks, int64_t skip)
    : skip_(skip) {
  for (int64_t i = 0; i < n_blocks; i++) {
    int64_t layer_input_dim = (i == 0 ? input_dim : hidden_dim);
    if (i == skip) {
      layer_input_dim += input_dim;
    }

    int64_t layer_output_dim = ((i == n_blocks - 1) ? output_dim : hidden_dim);
    layers_->push_back(torch::nn::Linear(layer_input_dim, layer_output_dim));
  }
  register_module("layers_", layers_);
}

torch::Tensor BasicMLPImpl::forward(torch::Tensor input) {
  torch::Tensor x = input;
  for (int64_t i = 0; i < layers_->size(); i++) {
    if (i == skip_) {
      x = torch::cat({x, input}, -1);
    }
    x = layers_[i]->as<torch::nn::Linear>()->forward(x);
    if (i != layers_->size() - 1) {
      x = torch::relu(x);
    }
  }
  return x;
}

VoxelBasedRadianceFieldImpl::VoxelBasedRadianceFieldImpl() {
  // Assuming fixed values for grids_lens_ and bound_.
  grids_lens_ = {0.64, 0.48, 0.32, 0.24, 0.16, 0.12, 0.08};
  bound_ = torch::tensor({{-1, 1}, {-1, 1}, {-1, 1}});
  const int grids_dim = 4;
  torch::Tensor xyz_len = (bound_.index({Slice(), 1}) - bound_.index({Slice(), 0})).to(torch::kFloat);

  for (int64_t i = 0; i < grids_lens_.size(); i++) {
    const double grid_len = grids_lens_[i];
    std::vector<int64_t> grid_shape;
    grid_shape.push_back(xyz_len[0].item<float>() / grid_len);
    grid_shape.push_back(xyz_len[1].item<float>() / grid_len);
    grid_shape.push_back(xyz_len[2].item<float>() / grid_len);
    std::swap(grid_shape[0], grid_shape[2]);

    std::vector<int64_t> grid_true_shape = {1, grids_dim};
    grid_true_shape.insert(grid_true_shape.end(), grid_shape.begin(), grid_shape.end());
    torch::Tensor grid = torch::zeros(grid_true_shape).set_requires_grad(true);

    grids_feat_.push_back(grid);
    grids_shape_.push_back(grid_shape);

    register_parameter("grid_feat_" + std::to_string(i), grids_feat_[i]);
  }

  // Assume args.nerf.decoder is "basic_MLP"
  alpha_decoder_ = BasicMLP(static_cast<int64_t>(grids_lens_.size()),
                            static_cast<int64_t>(256),  // hidden_dim,
                            static_cast<int64_t>(1),    // output_dim,
                            static_cast<int64_t>(5),    // n_blocks,
                            static_cast<int64_t>(2)     // skip,
  );
  color_decoder_ = BasicMLP(static_cast<int64_t>(grids_lens_.size() * 3),
                            static_cast<int64_t>(256),  // hidden_dim,
                            static_cast<int64_t>(3),    // output_dim,
                            static_cast<int64_t>(5),    // n_blocks,
                            static_cast<int64_t>(2)     // skip,
  );
  register_module("alpha_decoder_", alpha_decoder_);
  register_module("color_decoder_", color_decoder_);
}

torch::Tensor VoxelBasedRadianceFieldImpl::normalize_3d_coordinate(torch::Tensor p, double grid_len,
                                                                   std::vector<int64_t> grid_shape) {
  torch::Tensor grid_xyz = torch::tensor(grid_shape).to(torch::kFloat) * grid_len;
  p = p.view({-1, 3});
  p.index({Slice(), 0}) = ((p.index({Slice(), 0}) - bound_.index({0, 0})) / grid_xyz[2]) * 2 - 1.0;
  p.index({Slice(), 1}) = ((p.index({Slice(), 1}) - bound_.index({1, 0})) / grid_xyz[1]) * 2 - 1.0;
  p.index({Slice(), 2}) = ((p.index({Slice(), 2}) - bound_.index({2, 0})) / grid_xyz[0]) * 2 - 1.0;
  return p;
}

std::pair<torch::Tensor, torch::Tensor> VoxelBasedRadianceFieldImpl::forward(torch::Tensor x, torch::Tensor d) {
  std::vector<torch::Tensor> raw_alpha_input;
  std::vector<torch::Tensor> raw_color_input;

  for (int64_t i = 0; i < grids_feat_.size(); i++) {
    torch::Tensor p_norm = normalize_3d_coordinate(x.clone(), grids_lens_[i], grids_shape_[i]).unsqueeze(0);
    p_norm = p_norm.index({Slice(), Slice(), None, None}).to(torch::kFloat);
    torch::Tensor c = torch::grid_sampler(grids_feat_[i],  // input
                                          p_norm,          // grid_points
                                          0,               // mode='bilinear', padding_mode='zeros'
                                          false,           // align_corners
                                          false            // grid_sampler
                                          )
                          .squeeze(-1)
                          .squeeze(-1)
                          .transpose(1, 2)
                          .squeeze(0);
    raw_alpha_input.push_back(c.index({Slice(), -1}).view({-1, 1}));
    raw_color_input.push_back(c.index({Slice(), Slice(None, 3)}));
  }

  torch::Tensor raw_alpha_input_cat = torch::cat(raw_alpha_input, -1);
  torch::Tensor raw_color_input_cat = torch::cat(raw_color_input, -1);

  torch::Tensor alpha = alpha_decoder_->forward(raw_alpha_input_cat);
  torch::Tensor color = color_decoder_->forward(raw_color_input_cat);
  return {color, alpha};
}
