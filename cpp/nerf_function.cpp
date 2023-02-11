#include "nerf_function.hpp"

#include <random>

Partition SplitRay(float t_n, float t_f, int32_t N, int32_t batch_size) {
  const float width = t_f - t_n;
  const float unit_width = width / N;
  std::vector<float> partition;
  partition.push_back(t_n);
  for (int32_t i = 0; i < N; i++) {
    partition.push_back(t_n + unit_width * (i + 1));
  }
  Partition result(batch_size, partition);
  return result;
}

Vec2D<float> SampleCoarse(const Partition& partition) {
  const int32_t batch_size = partition.size();
  Vec2D<float> result(batch_size);
  std::mt19937_64 engine(std::random_device{}());
  for (int32_t i = 0; i < batch_size; i++) {
    const int32_t M = partition[i].size();
    result[i].resize(M - 1);
    for (int64_t j = 0; j < M - 1; j++) {
      std::uniform_real_distribution<float> dist(partition[i][j], partition[i][j + 1]);
      result[i][j] = dist(engine);
    }
  }
  return result;
}

Vec2D<float> _pcpdf(const Partition& partition, Vec2D<float> weights, int32_t N_s) {
  const int32_t batch_size = weights.size();
  const int32_t N_p = weights.front().size();

  // normalize weights
  for (int32_t i = 0; i < batch_size; i++) {
    float sum = 0;
    for (int32_t j = 0; j < N_p; j++) {
      constexpr float kEps = 1e-16;
      weights[i][j] = std::max(weights[i][j], kEps);
      sum += weights[i][j];
    }
    for (int32_t j = 0; j < N_p; j++) {
      weights[i][j] /= sum;
    }
  }

  std::mt19937_64 engine(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  Vec2D<float> _sample(batch_size, std::vector<float>(N_p));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N_p; j++) {
      _sample[i][j] = dist(engine);
    }
    sort(_sample[i].begin(), _sample[i].end());
  }

  // Slopes of a piecewise linear function
  Vec2D<float> a(batch_size, std::vector<float>(N_p));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N_p; j++) {
      a[i][j] = (partition[i][j + 1] - partition[i][j]) / weights[i][j];
    }
  }

  // Intercepts of a piecewise linear function
  Vec2D<float> cum_weights(batch_size, std::vector<float>(N_p + 1, 0));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N_p; j++) {
      cum_weights[i][j + 1] = cum_weights[i][j] + weights[i][j];
    }
  }

  Vec2D<float> b(batch_size, std::vector<float>(N_p));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N_p; j++) {
      b[i][j] = partition[i][j] - a[i][j] * cum_weights[i][j];
    }
  }

  Vec2D<float> sample(batch_size, std::vector<float>(N_p, 0));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N_p; j++) {
      const float min_j = cum_weights[i][j];
      const float max_j = cum_weights[i][j + 1];
      const bool mask = ((min_j <= _sample[i][j]) && (_sample[i][j] < max_j));
      sample[i][j] += (a[i][j] * _sample[i][j] + b[i][j]) * mask;
    }
  }

  return sample;
}

Vec2D<float> SampleFine(const Partition& partition, const Vec2D<float>& weights, const Vec2D<float>& t_c, int32_t N_f) {
  Vec2D<float> t_f = _pcpdf(partition, weights, N_f);
  const int32_t batch_size = t_c.size();
  Vec2D<float> t(batch_size);
  for (int32_t i = 0; i < batch_size; i++) {
    t[i].insert(t[i].end(), t_c[i].begin(), t_c[i].end());
    t[i].insert(t[i].end(), t_f[i].begin(), t_f[i].end());
    sort(t[i].begin(), t[i].end());
  }
  return t;
}

Vec3D<float> MakeRay(const Vec2D<float>& o, const Vec2D<float>& d, const Vec2D<float>& t) {
  const int32_t batch_size = o.size();
  const int32_t N = t.front().size();
  Vec3D<float> result(batch_size, Vec2D<float>(N, std::vector<float>(3)));
  for (int32_t i = 0; i < batch_size; i++) {
    for (int32_t j = 0; j < N; j++) {
      for (int32_t k = 0; k < 3; k++) {
        result[i][j][k] = t[i][j] * d[i][k] + o[i][k];
      }
    }
  }
  return result;
}

std::pair<RGB, Weight> _rgb_and_weight(RadianceField func, const Vec2D<float>& o, const Vec2D<float>& d,
                                       const Vec2D<float>& t, int32_t N) {
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
