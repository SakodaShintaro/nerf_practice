import numpy as np
from PIL import Image
from make_dataset import get_camera_intrinsic_parameter, get_dataset_raw
from nerf_model import NeRF
import torch
import os
from constants import DATASET_PATH, RESULT_DIR
from tqdm import tqdm
from sample_function import camera_params_to_rays


if __name__ == "__main__":
    print(DATASET_PATH)
    param = get_camera_intrinsic_parameter(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)

    # 512 * 512 はやや時間がかかるので半分のサイズでレンダリング
    param.f /= 2
    param.cx /= 2
    param.cy /= 2
    param.width //= 2
    param.height //= 2

    nerf = NeRF(t_n=0., t_f=2.5, c_bg=(1, 1, 1))
    nerf.load_state_dict(torch.load(f"{RESULT_DIR}/train/nerf_model.pt"))
    nerf.to("cuda")
    nerf.eval()

    images = []

    save_dir = f"{RESULT_DIR}/result_images"
    os.makedirs(save_dir, exist_ok=True)

    for ind, a in enumerate(tqdm(np.linspace(-np.pi, np.pi, 65)[:-1])):
        c = np.cos(a)
        s = np.sin(a)

        # y軸回り
        R = np.array([[c, 0, -s, 0],
                      [0, 1, 0, 0],
                      [s, 0, c, 0],
                      [0, 0, 0, 1]], dtype=np.float32)

    #     # z軸回り
    #     R = np.array([[c, -s, 0, 0],
    #                   [s,  c, 0, 0],
    #                   [0,  0, 1, 0],
    #                   [0,  0, 0, 1]], dtype=np.float32)

        o, d = camera_params_to_rays(param, R @ dataset_raw[200]["pose"])
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)

        _C_c = []
        _C_f = []
        with torch.no_grad():
            for i in range(0, o.shape[0], nerf.BATCH_SIZE):
                o_i = o[i:i + nerf.BATCH_SIZE]
                d_i = d[i:i + nerf.BATCH_SIZE]
                C_c_i, C_f_i = nerf.forward(o_i, d_i)
                _C_c.append(C_c_i.cpu().numpy())
                _C_f.append(C_f_i.cpu().numpy())

        C_c = np.concatenate(_C_c, axis=0)
        C_f = np.concatenate(_C_f, axis=0)
        C_c = np.clip(0., 1., C_c.reshape(param.height, param.width, 3))
        C_f = np.clip(0., 1., C_f.reshape(param.height, param.width, 3))

        image = Image.fromarray((C_f * 255.).astype(np.uint8))
        image.save(f"{save_dir}/{ind:08d}.png")
        images.append(image)

    # GIF版
    images[0].save(f'{save_dir}/greek.gif', save_all=True, append_images=images[1:],
                   duration=125, loop=0)
