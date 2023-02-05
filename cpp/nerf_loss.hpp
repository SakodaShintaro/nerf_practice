#ifndef NERF_LOSS_HPP
#define NERF_LOSS_HPP

#include "nerf_function.hpp"

class NeRFLoss : public torch::nn::Module {
 public:
  NeRFLoss(NeRF nerf);
  torch::Tensor forward(const Vec2D<float>& o, const Vec2D<float>& d, const Vec2D<float>& C);

 private:
  NeRF nerf_;
};

#endif
