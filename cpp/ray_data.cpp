#include "ray_data.hpp"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> RayData2Tensor(const std::vector<RayData>& data) {
  std::vector<float> o_vec;
  std::vector<float> d_vec;
  std::vector<float> C_vec;
  for (const RayData& datum : data) {
    o_vec.push_back(datum.o.x());
    o_vec.push_back(datum.o.y());
    o_vec.push_back(datum.o.z());
    d_vec.push_back(datum.d.x());
    d_vec.push_back(datum.d.y());
    d_vec.push_back(datum.d.z());
    C_vec.push_back(datum.bgr[0]);
    C_vec.push_back(datum.bgr[1]);
    C_vec.push_back(datum.bgr[2]);
  }
  torch::Tensor o = torch::tensor(o_vec).view({-1, 3});
  torch::Tensor d = torch::tensor(d_vec).view({-1, 3});
  torch::Tensor C = torch::tensor(C_vec).view({-1, 3});
  return std::make_tuple(o, d, C);
}
