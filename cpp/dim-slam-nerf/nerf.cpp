#include "nerf.hpp"

using namespace torch::nn;

BasicMLPImpl::BasicMLPImpl(int64_t input_dim, int64_t hidden_dim, int64_t output_dim, int64_t n_blocks, int64_t skip)
    : skip_(skip) {
  for (int64_t i = 0; i < n_blocks; i++) {
    int64_t layer_input_dim = (i == 0 ? input_dim : hidden_dim);
    if (i == skip) {
      layer_input_dim += input_dim;
    }

    int64_t layer_output_dim = ((i == n_blocks - 1) ? output_dim : hidden_dim);
    layers_->push_back(Linear(layer_input_dim, layer_output_dim));
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
