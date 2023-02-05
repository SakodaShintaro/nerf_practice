#ifndef NERF_FUNCTION_HPP
#define NERF_FUNCTION_HPP

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <vector>

#include "view.hpp"

using Position = std::array<float, 3>;

template <class T>
using Vec2D = std::vector<std::vector<T>>;
template <class T>
using Vec3D = std::vector<Vec2D<T>>;

using Partition = Vec2D<float>;

using RGB = std::array<float, 3>;
using Weight = float;

Partition SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size);
Vec2D<float> SampleCoarse(const Partition& partition);
Vec2D<float> _pcpdf(const Partition& partition, const Vec2D<float>& weights, int32_t N_s);
Vec2D<float> SampleFine(const Partition& partition, const Vec2D<float>& weights, const Vec2D<float>& t_c, int32_t N_f);
Vec3D<float> Ray(const Vec2D<float>& o, const Vec2D<float>& d, const Vec2D<float>& t);
std::pair<RGB, Weight> _rgb_and_weight(torch::nn::Module func, const Vec2D<float>& o, const Vec2D<float>& d,
                                       const Vec2D<float>& t, int32_t N);
std::pair<torch::Tensor, torch::Tensor> VolumeRenderingWithRadianceField(
    torch::nn::Module func_c, torch::nn::Module func_f, const Vec2D<float>& o, const Vec2D<float>& d,
    const Vec2D<float>& t_n, const Vec2D<float>& t_f, int32_t N_c, int32_t N_f, const RGB& c_bg);
std::pair<Vec2D<Position>, Vec3D<Position>> CameraParamsToRays(const View& view);

#endif
