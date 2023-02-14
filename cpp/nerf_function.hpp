#ifndef NERF_FUNCTION_HPP
#define NERF_FUNCTION_HPP

#include <torch/torch.h>

#include <Eigen/Core>
#include <array>
#include <cstdint>
#include <vector>

#include "camera_intrinsic_parameter.hpp"
#include "data.hpp"
#include "radiance_field.hpp"
#include "ray_data.hpp"

torch::Tensor SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size);
torch::Tensor SampleCoarse(torch::Tensor partition);
torch::Tensor _pcpdf(torch::Tensor partition, torch::Tensor weights, int32_t N_s);
torch::Tensor SampleFine(torch::Tensor partition, torch::Tensor weights, torch::Tensor t_c, int32_t N_f);
torch::Tensor MakeRay(torch::Tensor o, torch::Tensor d, torch::Tensor t);
std::pair<torch::Tensor, torch::Tensor> _rgb_and_weight(RadianceField func, torch::Tensor o, torch::Tensor d,
                                                        torch::Tensor t, int32_t N);
std::vector<RayData> GetRays(const CameraIntrinsicParameter& param, const Data& data);

#endif
