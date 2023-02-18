from make_dataset import get_camera_intrinsic_parameter, get_dataset_raw
from constants import DATASET_PATH
from nerf_model import NeRF
import torch
from nerf_function import camera_params_to_rays, split_ray, sample_coarse, rgb_and_weight, sample_fine


def test():
    param = get_camera_intrinsic_parameter(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)
    i = 0
    pose = dataset_raw[i]['pose']
    rgb = dataset_raw[i]['rgb']
    o, d = camera_params_to_rays(param, pose)
    o = o.reshape(-1, 3)
    d = d.reshape(-1, 3)
    o = o[0:1]
    d = d[0:1]

    nerf = NeRF()
    device = nerf.device()
    o = torch.tensor(o, device=device)
    d = torch.tensor(d, device=device)
    print(f"o = {o}")
    print(f"d = {d}")

    batch_size = o.shape[0]
    device = o.device

    partitions = split_ray(nerf.t_n, nerf.t_f, nerf.N_c, batch_size)
    partitions = partitions.to(device)
    partitions = partitions.detach()
    print(f"partitions = {partitions}")

    # background.
    bg = torch.tensor(nerf.c_bg, device=device, dtype=torch.float32)
    bg = bg.view(1, 3)

    # coarse rendering.
    _t_c = sample_coarse(partitions)
    t_c = _t_c.to(device)
    print(f"t_c = {t_c}")

    rgb_c, w_c = rgb_and_weight(nerf.rf_c, o, d, t_c, nerf.N_c)
    C_c = torch.sum(w_c[..., None] * rgb_c, axis=1)
    C_c += (1. - torch.sum(w_c, axis=1, keepdims=True)) * bg

    # fine rendering.
    _w_c = w_c.clone()
    t_f = sample_fine(partitions, _w_c, _t_c, nerf.N_f)
    t_f = t_f.to(device)

    rgb_f, w_f = rgb_and_weight(nerf.rf_f, o, d, t_f, nerf.N_f + nerf.N_c)
    C_f = torch.sum(w_f[..., None] * rgb_f, axis=1)
    C_f += (1. - torch.sum(w_f, axis=1, keepdims=True)) * bg


if __name__ == "__main__":
    test()
