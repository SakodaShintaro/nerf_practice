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

template <class T>
using Vec2D = std::vector<std::vector<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

using Partition = Vec2D<float>;

using RGB = std::array<float, 3>;
using Weight = float;

Partition SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size);
torch::Tensor SampleCoarse(const Partition& partition);
Vec2D<float> _pcpdf(const Partition& partition, torch::Tensor weights, int32_t N_s);
torch::Tensor SampleFine(const Partition& partition, torch::Tensor weights, torch::Tensor t_c, int32_t N_f);
Vec3D<float> MakeRay(const Vec2D<float>& o, const Vec2D<float>& d, const Vec2D<float>& t);
std::pair<torch::Tensor, torch::Tensor> _rgb_and_weight(RadianceField func, torch::Tensor o, torch::Tensor d,
                                                        torch::Tensor t, int32_t N);
std::pair<torch::Tensor, torch::Tensor> VolumeRenderingWithRadianceField(
    torch::nn::Module func_c, torch::nn::Module func_f, const Vec2D<float>& o, const Vec2D<float>& d,
    const Vec2D<float>& t_n, const Vec2D<float>& t_f, int32_t N_c, int32_t N_f, const RGB& c_bg);
std::vector<RayData> GetRays(const CameraIntrinsicParameter& param, const Data& data);

#endif
