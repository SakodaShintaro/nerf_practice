#ifndef NERF_FUNCTION_HPP
#define NERF_FUNCTION_HPP

#include <torch/torch.h>

#include <array>
#include <cstdint>
#include <vector>

using Vec2D = std::vector<std::vector<float>>;
using Vec3D = std::vector<Vec2D>;
using Partition = std::vector<std::vector<float>>;

using RGB = std::array<float, 3>;
using Weight = float;

Partition SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size);
Vec2D SampleCoarse(const Partition& partition);
Vec2D _pcpdf(const Partition& partition, const Vec2D& weights, int32_t N_s);
Vec2D SampleFine(const Partition& partition, const Vec2D& weights, const Vec2D& t_c, int32_t N_f);
Vec3D Ray(const Vec2D& o, const Vec2D& d, const Vec2D& t);
std::pair<RGB, Weight> _rgb_and_weight(torch::nn::Module func, const Vec2D& o, const Vec2D& d, const Vec2D& t,
                                       int32_t N);
std::pair<torch::Tensor, torch::Tensor> VolumeRenderingWithRadianceField(torch::nn::Module func_c,
                                                                         torch::nn::Module func_f, const Vec2D& o,
                                                                         const Vec2D& d, const Vec2D& t_n,
                                                                         const Vec2D& t_f, int32_t N_c, int32_t N_f,
                                                                         const RGB& c_bg);

#endif
