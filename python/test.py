from make_dataset import get_camera_intrinsic_parameter, get_dataset_raw
from constants import DATASET_PATH
from nerf_model import NeRF
import torch
from nerf_function import camera_params_to_rays, split_ray, sample_coarse, rgb_and_weight, sample_fine


def test() -> None:
    param = get_camera_intrinsic_parameter(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)
    i = 0
    pose = dataset_raw[i].pose
    o_np, d_np = camera_params_to_rays(param, pose)
    o_np = o_np.reshape(-1, 3)
    d_np = d_np.reshape(-1, 3)
    o_np = o_np[0:1]
    d_np = d_np[0:1]

    nerf = NeRF()
    device = nerf.device()
    o_tensor = torch.tensor(o_np, device=device)
    d_tensor = torch.tensor(d_np, device=device)
    print(f"o = {o_tensor}")
    print(f"d = {d_tensor}")

    C_c, C_f = nerf.forward(o_tensor, d_tensor)
    print(f"C_c = {C_c}")
    print(f"C_f = {C_f}")

if __name__ == "__main__":
    test()
