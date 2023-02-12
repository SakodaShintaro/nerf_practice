import glob
import os
import numpy as np
from PIL import Image
from sample_function import camera_params_to_rays
from constants import DATASET_PATH, RESULT_DIR
from tqdm import tqdm
from camera_intrinsic_parameter import CameraIntrinsicParameter


def get_camera_intrinsic_parameter(dataset_path: str) -> CameraIntrinsicParameter:
    def _line2floats(line):
        return map(float, line.strip().split())

    with open(os.path.join(dataset_path, 'intrinsics.txt'), 'r') as file:
        # focal length, image centers.
        f, cx, cy, _ = _line2floats(file.readline())

        # origin
        origin_x, origin_y, origin_z = _line2floats(file.readline())

        # near plane
        near_plane, = _line2floats(file.readline())

        # scale
        scale, = _line2floats(file.readline())

        # image size
        img_height, img_width = _line2floats(file.readline())

    # データセットの画像サイズ．
    width = 512
    height = 512

    f = f * height / img_height
    cx = cx * width / img_width
    cy = cy * height / img_height

    return CameraIntrinsicParameter(f, cx, cy, width, height)


def get_dataset_raw(dataset_path: str):
    pose_paths = sorted(glob.glob(dataset_path + 'pose/*.txt'))
    rgb_paths = sorted(glob.glob(dataset_path + 'rgb/*.png'))

    dataset_raw = []

    for pose_path, rgb_path in zip(pose_paths, rgb_paths):
        pose = np.genfromtxt(
            pose_path, dtype=np.float32).reshape(4, 4)

        rgb = Image.open(rgb_path)

        data = {
            'pose': pose,
            'rgb': rgb,
        }
        dataset_raw.append(data)
    return dataset_raw


if __name__ == "__main__":
    param = get_camera_intrinsic_parameter(DATASET_PATH)
    dataset_raw = get_dataset_raw(DATASET_PATH)

    o_list = []
    d_list = []
    C_list = []

    for data in tqdm(dataset_raw):
        pose = data['pose']
        rgb = data['rgb']

        o, d = camera_params_to_rays(param, pose)
        C = (np.array(rgb, dtype=np.float32) / 255.)[:, :, :3]

        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        C = C.reshape(-1, 3)

        o_list.append(o)
        d_list.append(d)
        C_list.append(C)

    o_list = np.concatenate(o_list)
    d_list = np.concatenate(d_list)
    C_list = np.concatenate(C_list)

    dataset = {'o': o_list, 'd': d_list, 'C': C_list}

    # 保存しておく
    os.makedirs(RESULT_DIR, exist_ok=True)
    np.savez(f'{RESULT_DIR}/dataset.npz', **dataset)
