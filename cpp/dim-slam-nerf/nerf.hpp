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
  NeRFImpl() {
    // Assuming fixed values for grids_lens_ and bound_.
    grids_lens_ = {0.64, 0.48, 0.32, 0.24, 0.16, 0.12, 0.08};
    bound_ = torch::tensor({{-1, 1}, {-1, 1}, {1, 1}});
    grids_dim_ = 4;  // This value should be obtained from self.args.nerf.grids_dim_.
    torch::Tensor xyz_len_ =
        (bound_.index({torch::indexing::Slice(), 1}) - bound_.index({torch::indexing::Slice(), 0})).to(torch::kFloat);

    std::vector<float> xyz_len(xyz_len_.data_ptr<float>(), xyz_len_.data_ptr<float>() + xyz_len_.numel());

    for (int64_t i = 0; i < grids_lens_.size(); i++) {
      const double grid_len = grids_lens_[i];
      std::vector<int64_t> grid_shape;
      grid_shape.push_back(xyz_len[0] / grid_len);
      grid_shape.push_back(xyz_len[1] / grid_len);
      grid_shape.push_back(xyz_len[2] / grid_len);
      std::swap(grid_shape[0], grid_shape[2]);

      std::vector<int64_t> grid_true_shape = {1, grids_dim_};
      grid_true_shape.insert(grid_true_shape.end(), grid_shape.begin(), grid_shape.end());
      torch::Tensor grid = torch::zeros(grid_true_shape).set_requires_grad(true);

      grids_feat_.push_back(grid);
      grids_shape_.push_back(grid_shape);
      grids_xyz_.push_back(torch::tensor(grid_shape).to(torch::kFloat) * grid_len);
    }

    // Assume args.nerf.decoder is "basic_MLP"
    alpha_decoder_ = BasicMLP(static_cast<int64_t>(grids_lens_.size()),
                              static_cast<int64_t>(256),  // hidden_dim,
                              static_cast<int64_t>(1),    // output_dim,
                              static_cast<int64_t>(5),    // n_blocks,
                              static_cast<int64_t>(2)     // skip,
    );
    color_decoder_ = BasicMLP(static_cast<int64_t>(6),
                              static_cast<int64_t>(256),  // hidden_dim,
                              static_cast<int64_t>(3),    // output_dim,
                              static_cast<int64_t>(5),    // n_blocks,
                              static_cast<int64_t>(2)     // skip,
    );
    register_module("alpha_decoder_", alpha_decoder_);
    register_module("color_decoder_", color_decoder_);
  }

  torch::Tensor normalize_3d_coordinate(torch::Tensor p, double grid_len, std::vector<int64_t> grid_shape) {
    auto grid_xyz = torch::tensor(grid_shape).to(torch::kFloat) * grid_len;
    p = p.view({-1, 3});
    p.index({torch::indexing::Slice(), 0}) =
        ((p.index({torch::indexing::Slice(), 0}) - bound_.index({0, 0})) / grid_xyz[2]) * 2 - 1.0;
    p.index({torch::indexing::Slice(), 1}) =
        ((p.index({torch::indexing::Slice(), 1}) - bound_.index({1, 0})) / grid_xyz[1]) * 2 - 1.0;
    p.index({torch::indexing::Slice(), 2}) =
        ((p.index({torch::indexing::Slice(), 2}) - bound_.index({2, 0})) / grid_xyz[0]) * 2 - 1.0;
    return p;
  }

  torch::Tensor forward(torch::Tensor input) {
    torch::Tensor p = input.clone();

    std::vector<torch::Tensor> raw_alpha_input;
    std::vector<torch::Tensor> raw_color_input;

    for (int64_t i = 0; i < grids_feat_.size(); i++) {
      auto p_norm = normalize_3d_coordinate(p.clone(), grids_lens_[i], grids_shape_[i]).unsqueeze(0);
      auto c = torch::grid_sampler(grids_feat_[i],
                                   p_norm
                                       .index({torch::indexing::Slice(), torch::indexing::Slice(),
                                               torch::indexing::None, torch::indexing::None})
                                       .to(torch::kFloat),
                                   0,      // mode='bilinear', padding_mode='zeros'
                                   false,  // align_corners
                                   false   // grid_sampler
                                   )
                   .squeeze(-1)
                   .squeeze(-1)
                   .transpose(1, 2)
                   .squeeze(0);
      raw_alpha_input.push_back(c.index({torch::indexing::Slice(), -1}).view({-1, 1}));
      raw_color_input.push_back(c.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, 3)}));
    }

    torch::Tensor raw_alpha_input_cat = torch::cat(raw_alpha_input, -1);
    torch::Tensor raw_color_input_cat = torch::cat(raw_color_input, -1);

    torch::Tensor alpha = alpha_decoder_->forward(raw_alpha_input_cat);
    torch::Tensor color = color_decoder_->forward(raw_color_input_cat);
    return torch::cat({color, alpha}, -1);
  }

 private:
  std::vector<torch::Tensor> grids_feat_;
  std::vector<std::vector<int64_t>> grids_shape_;
  std::vector<double> grids_lens_;
  torch::Tensor bound_;
  int grids_dim_;
  std::vector<torch::Tensor> grids_xyz_;
  BasicMLP alpha_decoder_ = nullptr;
  BasicMLP color_decoder_ = nullptr;
};

TORCH_MODULE(NeRF);

#endif
