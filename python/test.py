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

    nerf.forward(o, d)


if __name__ == "__main__":
    test()
