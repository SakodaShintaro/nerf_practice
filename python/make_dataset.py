import glob
import os
import numpy as np
from PIL import Image
from nerf_model import camera_params_to_rays


def get_view(dataset_path: str):
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

    print('focal length: {}'.format(f))
    print('image center: ({}, {})'.format(cx, cy))
    print('image size: ({}, {})'.format(img_width, img_height))

    # データセットの画像サイズ．
    width = 512
    height = 512

    f = f * height / img_height
    cx = cx * width / img_width
    cy = cy * height / img_height

    print('focal length: {}'.format(f))
    print('image center: ({}, {})'.format(cx, cy))
    print('image size: ({}, {})'.format(width, height))
    return f, cx, cy, width, height


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


dataset_path = '/root/nerf_practice/data/train/greek/'


if __name__ == "__main__":
    f, cx, cy, width, height = get_view(dataset_path)
    dataset_raw = get_dataset_raw(dataset_path)

    os = []
    ds = []
    Cs = []

    for data in dataset_raw:
        pose = data['pose']
        rgb = data['rgb']

        o, d = camera_params_to_rays(f, cx, cy, pose, width, height)
        C = (np.array(rgb, dtype=np.float32) / 255.)[:, :, :3]

        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        C = C.reshape(-1, 3)

        os.append(o)
        ds.append(d)
        Cs.append(C)

    os = np.concatenate(os)
    ds = np.concatenate(ds)
    Cs = np.concatenate(Cs)

    dataset = {'o': os, 'd': ds, 'C': Cs}

    # 保存しておく
    np.savez('./dataset.npz', **dataset)
