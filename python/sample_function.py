import numpy as np
import torch
import torch.nn.functional as F
from camera_intrinsic_parameter import CameraIntrinsicParameter


def split_ray(t_n, t_f, N, batch_size):
    """Split the ray into N partitions.

    partition: [t_n, t_n + (1 / N) * (t_f - t_n), ..., t_f]

    Args:
        t_n (float): t_near. Start point of split.
        t_f (float): t_far. End point of split.
        N (int): Num of partitions.
        batch_size (int): Batch size.

    Returns:
        ndarray, [batch_size, N]: A partition.

    """
    partitions = np.linspace(t_n, t_f, N + 1, dtype=np.float32)
    return np.repeat(partitions[None], repeats=batch_size, axis=0)


def sample_coarse(partitions):
    """Sample ``t_i`` from partitions for ``coarse`` network.

    t_i ~ U[t_n + ((i - 1) / N) * (t_f - t_n), t_n + (i / N) * (t_f - t_n)]

    Args:
        partitions (ndarray, [batch_size, N+1]): Outputs of ``split_ray``.

    Return:
        ndarray, [batch_size, N]: Sampled t.

    """
    t = np.random.uniform(
        partitions[:, :-1], partitions[:, 1:]).astype(np.float32)
    return t


def _pcpdf(partitions, weights, N_s):
    """Sample from piecewise-constant probability density function.

    Args:
        partitions (ndarray, [batch_size, N_p+1]): N_p Partitions.
        weights (ndarray, [batch_size, N_p]): The ratio of sampling from each
            partition.
        N_s (int): Num of samples.

    Returns:
        numpy.ndarray, [batch_size, N_s]: Samples.

    """
    batch_size, N_p = weights.shape

    # normalize weights.
    weights[weights < 1e-16] = 1e-16
    weights /= weights.sum(axis=1, keepdims=True)

    _sample = np.random.uniform(
        0, 1, size=(batch_size, N_s)).astype(np.float32)
    _sample = np.sort(_sample, axis=1)

    # Slopes of a piecewise linear function.
    a = (partitions[:, 1:] - partitions[:, :-1]) / weights

    # Intercepts of a piecewise linear function.
    cum_weights = np.cumsum(weights, axis=1)
    cum_weights = np.pad(cum_weights, ((0, 0), (1, 0)),
                         mode='constant')
    b = partitions[:, :-1] - a * cum_weights[:, :-1]

    sample = np.zeros_like(_sample)
    for j in range(N_p):
        min_j = cum_weights[:, j:j + 1]
        max_j = cum_weights[:, j + 1:j + 2]
        a_j = a[:, j:j + 1]
        b_j = b[:, j:j + 1]
        mask = ((min_j <= _sample) & (_sample < max_j)).astype(np.float32)
        sample += (a_j * _sample + b_j) * mask

    return sample


def sample_fine(partitions, weights, t_c, N_f):
    """Sample ``t_i`` from partitions for ``fine`` network.

    Sampling from each partition according to given weights.

    Args:
        partitions (ndarray, [batch_size, N_c+1]): Outputs of ``split_ray``.
        weights (ndarray, [batch_size, N_c]):
            T_i * (1 - exp(- sigma_i * delta_i)).
        t_c (ndarray, [batch_size, N_c]): ``t`` of coarse rendering.
        N_f (int): num of sampling.

    Return:
        ndarray, [batch_size, N_c+N_f]: Sampled t.

    """
    t_f = _pcpdf(partitions, weights, N_f)
    t = np.concatenate([t_c, t_f], axis=1)
    t = torch.tensor(t)
    t, _ = torch.sort(t, dim=1)
    return t


def ray(o, d, t):
    """Returns points on the ray.

    Args:
        o (ndarray, [batch_size, 3]): Start points of the ray.
        d (ndarray, [batch_size, 3]): Directions of the ray.
        t (ndarray, [batch_size, N]): Sampled t.

    Returns:
        ndarray, [batch_size, N, 3]: Points on the ray.

    """
    return o[:, None] + t[..., None] * d[:, None]


def rgb_and_weight(func, o, d, t, N):
    batch_size = o.shape[0]

    x = ray(o, d, t)
    x = x.view(batch_size, N, -1)
    d = d[:, None].repeat(1, N, 1)

    x = x.view(batch_size * N, -1)
    d = d.view(batch_size * N, -1)

    # forward.
    rgb, sigma = func(x, d)

    rgb = rgb.view(batch_size, N, -1)
    sigma = sigma.view(batch_size, N, -1)

    delta = F.pad(t[:, 1:] - t[:, :-1], (0, 1), mode='constant', value=1e8)
    mass = sigma[..., 0] * delta
    mass = F.pad(mass, (1, 0), mode='constant', value=0.)

    alpha = 1. - torch.exp(- mass[:, 1:])
    T = torch.exp(- torch.cumsum(mass[:, :-1], dim=1))
    w = T * alpha
    return rgb, w


def camera_params_to_rays(param: CameraIntrinsicParameter, pose: np.ndarray):
    """Make rays (o, d) from camera parameters.

    Args:
        param : CameraIntrinsicParameter.
        pose (ndarray, [4, 4]): camera extrinsic matrix.

    Returns:
        o (ndarray, [height, width, 3]): The origin of the camera coordinate.
        d (ndarray, [height, width, 3]): The direction of each ray.

    """
    _o = np.zeros((param.height, param.width, 4), dtype=np.float32)
    _o[:, :, 3] = 1

    v, u = np.mgrid[:param.height, :param.width].astype(np.float32)
    _x = (u - param.cx) / param.f
    _y = (v - param.cy) / param.f
    _z = np.ones_like(_x)
    _w = np.ones_like(_x)
    _d = np.stack([_x, _y, _z, _w], axis=2)

    o = (pose @ _o[..., None])[..., :3, 0]
    _d = (pose @ _d[..., None])[..., :3, 0]
    d = _d - o
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return o, d
