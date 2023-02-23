#include "nerf_function.hpp"

#include <random>

torch::Tensor SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size) {
  const float width = t_f - t_n;
  const float unit_width = width / N;
  std::vector<float> partition;
  partition.push_back(t_n);
  for (int32_t i = 0; i < N; i++) {
    partition.push_back(t_n + unit_width * (i + 1));
  }
  torch::Tensor result = torch::tensor(partition);
  result = result.view({1, N + 1});
  result = result.repeat({batch_size, 1});
  return result;
}

torch::Tensor SampleCoarse(torch::Tensor partition) {
  using namespace torch::indexing;
  torch::Tensor l = partition.index({Slice(None, None), Slice(None, -1)});
  torch::Tensor r = partition.index({Slice(None, None), Slice(1, None)});
  torch::Tensor t = (r - l) * torch::rand_like(l) + l;
  return t;
}

torch::Tensor _pcpdf(torch::Tensor partition, torch::Tensor weights, int32_t N_s) {
  using namespace torch::indexing;
  const int32_t batch_size = weights.size(0);
  const int32_t N_p = weights.size(1);

  const torch::Tensor t = torch::tensor(1e-16f);
  weights = torch::maximum(weights, t);
  weights /= weights.sum(1, true);

  torch::Tensor _sample = torch::rand({batch_size, N_s}).to(weights.device());
  auto [s, i] = torch::sort(_sample, 1);
  _sample = s;

  torch::Tensor l = partition.index({Slice(None, None), Slice(None, -1)});
  torch::Tensor r = partition.index({Slice(None, None), Slice(1, None)});
  torch::Tensor a = (r - l) / weights;

  torch::Tensor cum_weights = torch::cumsum(weights, 1);
  cum_weights = torch::pad(cum_weights, {1, 0, 0, 0});

  torch::Tensor b = l - a * cum_weights.index({Slice(None, None), Slice(None, -1)});

  torch::Tensor sample = torch::zeros_like(_sample);

  for (int32_t j = 0; j < N_p; j++) {
    torch::Tensor min_j = cum_weights.index({Slice(None, None), Slice(j, j + 1)});
    torch::Tensor max_j = cum_weights.index({Slice(None, None), Slice(j + 1, j + 2)});
    torch::Tensor a_j = a.index({Slice(None, None), Slice(j, j + 1)});
    torch::Tensor b_j = b.index({Slice(None, None), Slice(j, j + 1)});
    torch::Tensor mask = ((min_j <= sample) & (_sample < max_j)).to(torch::kFloat32);
    sample += (a_j * _sample + b_j) * mask;
  }
  return sample;
}

torch::Tensor SampleFine(torch::Tensor partition, torch::Tensor weights, torch::Tensor t_c, int32_t N_f) {
  torch::Tensor t_f = _pcpdf(partition, weights, N_f);
  torch::Tensor t = torch::concatenate({t_c, t_f}, 1);
  auto [tt, _] = torch::sort(t, 1);
  return tt;
}

torch::Tensor MakeRay(torch::Tensor o, torch::Tensor d, torch::Tensor t) {
  const int32_t batch_size = o.size(0);
  const int32_t N = t.size(1);
  o = o.view({batch_size, 1, 3});
  d = d.view({batch_size, 1, 3});
  t = t.view({batch_size, N, 1});
  return o + t * d;
}

std::pair<torch::Tensor, torch::Tensor> _rgb_and_weight(RadianceField func, torch::Tensor o, torch::Tensor d,
                                                        torch::Tensor t, int32_t N) {
  using namespace torch::indexing;
  const int32_t batch_size = o.size(0);

  torch::Tensor x = MakeRay(o, d, t);
  x = x.view({batch_size, N, -1});

  d = d.unsqueeze(-1);
  d = d.repeat({1, N, 1});

  x = x.view({batch_size * N, -1});
  d = d.view({batch_size * N, -1});

  auto [rgb, sigma] = func(x, d);

  rgb = rgb.view({batch_size, N, -1});
  sigma = sigma.view({batch_size, N, -1});

  torch::Tensor tl = t.index({Slice(None, None), Slice(None, -1)});
  torch::Tensor tr = t.index({Slice(None, None), Slice(1, None)});
  torch::Tensor delta = torch::pad(tr - tl, {0, 1}, "constant", 1e8);
  torch::Tensor mass = sigma.index({"...", 0}) * delta;
  mass = torch::pad(mass, {1, 0});

  torch::Tensor mass_l = mass.index({Slice(None, None), Slice(None, -1)});
  torch::Tensor mass_r = mass.index({Slice(None, None), Slice(1, None)});
  torch::Tensor alpha = 1 - torch::exp(-mass_r);
  torch::Tensor T = torch::exp(-torch::cumsum(mass_l, 1));
  torch::Tensor w = T * alpha;
  return std::make_pair(rgb, w);
}

std::vector<RayData> GetRays(const CameraIntrinsicParameter& param, const Data& data) {
  const Position center = data.pose.block(0, 3, 3, 1);

  std::vector<RayData> rays;
  for (int32_t i = 0; i < param.height; i++) {
    for (int32_t j = 0; j < param.width; j++) {
      const float y = (i + 0.5f - param.cy) / param.f;
      const float x = (j + 0.5f - param.cx) / param.f;
      const float z = 1.0f;
      Eigen::Vector4f vec;
      vec << x, y, z, 1.0f;
      vec = data.pose * vec;
      Position d = vec.block(0, 0, 3, 1);
      d -= center;
      d.normalize();
      rays.push_back({center, d, data.image.at<cv::Vec3b>(i, j)});
    }
  }
  return rays;
}
