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
  return result;
}

torch::Tensor SampleCoarse(torch::Tensor partition) {
  const int32_t batch_size = partition.size(0);
  const int32_t M = partition.size(1);
  std::vector<float> result(batch_size * (M - 1));
  std::mt19937_64 engine(std::random_device{}());
  for (int32_t i = 0; i < batch_size; i++) {
    for (int64_t j = 0; j < M - 1; j++) {
      std::uniform_real_distribution<float> dist(partition[i][j].item<float>(), partition[i][j + 1].item<float>());
      result[i * (M - 1) + j] = dist(engine);
    }
  }
  return torch::tensor(result).view({batch_size, M - 1});
}

torch::Tensor _pcpdf(torch::Tensor partition, torch::Tensor weights, int32_t N_s) {
  const int32_t batch_size = weights.size(0);
  const int32_t N_p = weights.size(1);

  const torch::Tensor t = torch::tensor(1e-16f);
  weights = torch::maximum(weights, t);
  weights /= weights.sum(1, true);

  torch::Tensor _sample = torch::rand_like(weights);
  auto [s, i] = torch::sort(_sample, 1);
  _sample = s;

  // // normalize weights
  // for (int32_t i = 0; i < batch_size; i++) {
  //   float sum = 0;
  //   for (int32_t j = 0; j < N_p; j++) {
  //     constexpr float kEps = 1e-16;
  //     weights[i][j] = std::max(weights[i][j], kEps);
  //     sum += weights[i][j];
  //   }
  //   for (int32_t j = 0; j < N_p; j++) {
  //     weights[i][j] /= sum;
  //   }
  // }

  // std::mt19937_64 engine(std::random_device{}());
  // std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  // Vec2D<float> _sample(batch_size, std::vector<float>(N_p));
  // for (int32_t i = 0; i < batch_size; i++) {
  //   for (int32_t j = 0; j < N_p; j++) {
  //     _sample[i][j] = dist(engine);
  //   }
  //   sort(_sample[i].begin(), _sample[i].end());
  // }

  // // Slopes of a piecewise linear function
  // Vec2D<float> a(batch_size, std::vector<float>(N_p));
  // for (int32_t i = 0; i < batch_size; i++) {
  //   for (int32_t j = 0; j < N_p; j++) {
  //     a[i][j] = (partition[i][j + 1] - partition[i][j]) / weights[i][j];
  //   }
  // }

  // // Intercepts of a piecewise linear function
  // Vec2D<float> cum_weights(batch_size, std::vector<float>(N_p + 1, 0));
  // for (int32_t i = 0; i < batch_size; i++) {
  //   for (int32_t j = 0; j < N_p; j++) {
  //     cum_weights[i][j + 1] = cum_weights[i][j] + weights[i][j];
  //   }
  // }

  // Vec2D<float> b(batch_size, std::vector<float>(N_p));
  // for (int32_t i = 0; i < batch_size; i++) {
  //   for (int32_t j = 0; j < N_p; j++) {
  //     b[i][j] = partition[i][j] - a[i][j] * cum_weights[i][j];
  //   }
  // }

  // Vec2D<float> sample(batch_size, std::vector<float>(N_p, 0));
  // for (int32_t i = 0; i < batch_size; i++) {
  //   for (int32_t j = 0; j < N_p; j++) {
  //     const float min_j = cum_weights[i][j];
  //     const float max_j = cum_weights[i][j + 1];
  //     const bool mask = ((min_j <= _sample[i][j]) && (_sample[i][j] < max_j));
  //     sample[i][j] += (a[i][j] * _sample[i][j] + b[i][j]) * mask;
  //   }
  // }

  // return sample;
}

torch::Tensor SampleFine(torch::Tensor partition, torch::Tensor weights, torch::Tensor t_c, int32_t N_f) {
  // Vec2D<float> t_f = _pcpdf(partition, weights, N_f);
  // const int32_t batch_size = t_c.size(0);
  // std::vector<float> result;
  // for (int32_t i = 0; i < batch_size; i++) {
  //   std::vector<float> tmp;
  //   tmp.insert(tmp.end(), t_c[i].begin(), t_c[i].end());
  //   tmp.insert(tmp.end(), t_f[i].begin(), t_f[i].end());
  //   sort(tmp.begin(), tmp.end());
  //   result.insert(result.end(), tmp.begin(), tmp.end());
  // }
  // return torch::tensor(result).view({batch_size, -1});
  return torch::Tensor();
}

torch::Tensor MakeRay(torch::Tensor o, torch::Tensor d, torch::Tensor t) {
  const int32_t batch_size = o.size(0);
  const int32_t N = t.size(1);
  return o + t * d;
}

std::pair<torch::Tensor, torch::Tensor> _rgb_and_weight(RadianceField func, torch::Tensor o, torch::Tensor d,
                                                        torch::Tensor t, int32_t N) {
  // const int32_t batch_size = o.size();

  // Vec3D<float> x = Ray(o, d, t);
  // torch::Tensor x_tensor = torch::tensor(x);
  // torch::Tensor d_tensor = torch::tensor(d);
  // auto [rgb, sigma] = func.forward(x_tensor, d_tensor);

  // rgb = rgb.view({batch_size, N, -1});
  // sigma = sigma.view({batch_size, N, -1});

  // torch::Tensor delta = torch::pad(t[:, 1:] - t[:, :-1], (0, 1), mode = "constanta", value = 1e8);
  // torch::Tensor mass = sigma[..., 0] * delta;
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
