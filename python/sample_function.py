import numpy as np
import torch
import torch.nn.functional as F


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
    t = np.sort(t, axis=1)
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


def _rgb_and_weight(func, o, d, t, N):
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


def volume_rendering_with_radiance_field(func_c, func_f, o, d, t_n, t_f,
                                         N_c, N_f, c_bg):
    """Rendering with Neural Radiance Field.

    Args:
        func_c: NN for coarse rendering.
        func_f: NN for fine rendering.
        o (ndarray, [batch_size, 3]): Start points of the ray.
        d (ndarray, [batch_size, 3]): Directions of the ray.
        t_n (float): Start point of split.
        t_f (float): End point of split.
        N_c (int): num of coarse sampling.
        N_f (int): num of fine sampling.
        c_bg (tuple, [3,]): Background color.

    Returns:
        C_c (tensor, [batch_size, 3]): Result of coarse rendering.
        C_f (tensor, [batch_size, 3]): Result of fine rendering.

    """
    batch_size = o.shape[0]
    device = o.device

    partitions = split_ray(t_n, t_f, N_c, batch_size)

    # background.
    bg = torch.tensor(c_bg, device=device, dtype=torch.float32)
    bg = bg.view(1, 3)

    # coarse rendering:
    _t_c = sample_coarse(partitions)
    t_c = torch.tensor(_t_c)
    t_c = t_c.to(device)

    rgb_c, w_c = _rgb_and_weight(func_c, o, d, t_c, N_c)
    C_c = torch.sum(w_c[..., None] * rgb_c, axis=1)
    C_c += (1. - torch.sum(w_c, axis=1, keepdims=True)) * bg

    # fine rendering.
    _w_c = w_c.detach().cpu().numpy()
    t_f = sample_fine(partitions, _w_c, _t_c, N_f)
    t_f = torch.tensor(t_f)
    t_f = t_f.to(device)

    rgb_f, w_f = _rgb_and_weight(func_f, o, d, t_f, N_f + N_c)
    C_f = torch.sum(w_f[..., None] * rgb_f, axis=1)
    C_f += (1. - torch.sum(w_f, axis=1, keepdims=True)) * bg

    return C_c, C_f
