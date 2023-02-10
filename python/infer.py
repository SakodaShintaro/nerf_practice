import numpy as np
from PIL import Image
from make_dataset import get_view, get_dataset_raw
from nerf_model import NeRF
import torch
import os
from constants import DATASET_PATH, RESULT_DIR

if __name__=="__main__":
    print(DATASET_PATH)
    f, cx, cy, width, height = get_view(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)

    ind = 200
    pose = dataset_raw[ind]['pose']
    rgb = dataset_raw[ind]['rgb']

    # 512 * 512 はやや時間がかかるので半分のサイズでレンダリング
    view = {
        'f': f / 2,
        'cx': cy / 2,
        'cy': cy / 2,
        'height': height // 2,
        'width': width // 2,
        'pose': pose
    }

    # original size.
    # view = {
    #     'f': f,
    #     'cx': cy,
    #     'cy': cy,
    #     'height': height,
    #     'width': width,
    #     'pose': pose
    # }

    nerf = NeRF(t_n=0., t_f=2.5, c_bg=(1, 1, 1))
    nerf.load_state_dict(torch.load(f"{RESULT_DIR}/train/nerf_model.pt"))
    nerf.to("cuda")
    C_c, C_f = nerf(view)

    images = []

    save_dir = f"{RESULT_DIR}/result_images"
    os.makedirs(save_dir, exist_ok=True)

    for ind, a in enumerate(np.linspace(-np.pi, np.pi, 65)[:-1]):
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

        _view = view.copy()
        _view['pose'] = R @ view['pose']

        C_c, C_f = nerf(_view)

        image = Image.fromarray((C_f * 255.).astype(np.uint8))
        image.save(f"{save_dir}/{ind:08d}.png")
        images.append(image)

    # GIF版
    images[0].save(f'{save_dir}/greek.gif', save_all=True, append_images=images[1:],
                duration=125, loop=0)
