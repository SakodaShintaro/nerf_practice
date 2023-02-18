import numpy as np
import torch
import torch.nn.functional as F
from camera_intrinsic_parameter import CameraIntrinsicParameter


def split_ray(t_n: float, t_f: float, N: int, batch_size: int) -> torch.Tensor:
    """Split the ray into N partitions.

    partition: [t_n, t_n + (1 / N) * (t_f - t_n), ..., t_f]

    Args:
        t_n (float): t_near. Start point of split.
        t_f (float): t_far. End point of split.
        N (int): Num of partitions.
        batch_size (int): Batch size.

    Returns:
        Tensor, [batch_size, N]: A partition.

    """
    partitions = np.linspace(t_n, t_f, N + 1, dtype=np.float32)
    partitions = np.repeat(partitions[None], repeats=batch_size, axis=0)
    return torch.tensor(partitions)


def sample_coarse(partitions: torch.Tensor) -> torch.Tensor:
    """Sample ``t_i`` from partitions for ``coarse`` network.

    t_i ~ U[t_n + ((i - 1) / N) * (t_f - t_n), t_n + (i / N) * (t_f - t_n)]

    Args:
        partitions (Tensor, [batch_size, N+1]): Outputs of ``split_ray``.

    Return:
        Tensor, [batch_size, N]: Sampled t.

    """
    l = partitions[:, :-1]
    r = partitions[:, 1:]
    t = (r - l) * torch.rand_like(l) + l
    return t


def _pcpdf(partitions: torch.Tensor, weights: torch.Tensor, N_s: int) -> torch.Tensor:
    """Sample from piecewise-constant probability density function.

    Args:
        partitions (Tensor, [batch_size, N_p+1]): N_p Partitions.
        weights (Tensor, [batch_size, N_p]): The ratio of sampling from each
            partition.
        N_s (int): Num of samples.

    Returns:
        Tensor, [batch_size, N_s]: Samples.

    """
    batch_size, N_p = weights.shape

    # normalize weights.
    weights[weights < 1e-16] = 1e-16
    weights /= weights.sum(dim=1, keepdims=True)

    _sample = torch.rand((batch_size, N_s)).to(weights.device)
    _sample, _ = torch.sort(_sample, dim=1)

    # Slopes of a piecewise linear function.
    a = (partitions[:, 1:] - partitions[:, :-1]) / weights

    # Intercepts of a piecewise linear function.
    cum_weights = torch.cumsum(weights, dim=1)
    cum_weights = F.pad(cum_weights, (1, 0, 0, 0), mode='constant')
    b = partitions[:, :-1] - a * cum_weights[:, :-1]

    sample = torch.zeros_like(_sample)
    for j in range(N_p):
        min_j = cum_weights[:, j:j + 1]
        max_j = cum_weights[:, j + 1:j + 2]
        a_j = a[:, j:j + 1]
        b_j = b[:, j:j + 1]
        mask = ((min_j <= _sample) & (_sample < max_j)).to(torch.float32)
        sample += (a_j * _sample + b_j) * mask

    return sample


def sample_fine(partitions: torch.Tensor, weights: torch.Tensor, t_c: torch.Tensor, N_f: int) -> torch.Tensor:
    """Sample ``t_i`` from partitions for ``fine`` network.

    Sampling from each partition according to given weights.

    Args:
        partitions (Tensor, [batch_size, N_c+1]): Outputs of ``split_ray``.
        weights (Tensor, [batch_size, N_c]):
            T_i * (1 - exp(- sigma_i * delta_i)).
        t_c (Tensor, [batch_size, N_c]): ``t`` of coarse rendering.
        N_f (int): num of sampling.

    Return:
        Tensor, [batch_size, N_c+N_f]: Sampled t.

    """
    t_f = _pcpdf(partitions, weights, N_f)
    t = torch.concatenate([t_c, t_f], dim=1)
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
