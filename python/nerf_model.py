import numpy as np
import torch
import torch.nn as nn
from radiance_field import RadianceField
from sample_function import *


def camera_params_to_rays(f, cx, cy, pose, width, height):
    """Make rays (o, d) from camera parameters.

    Args:
        f (float): A focal length.
        cx, xy (float): A center of the image.
        pose (ndarray, [4, 4]): camera extrinsic matrix.
        width(int): The height of the rendered image.
        height(int): The width of the rendered image.

    Returns:
        o (ndarray, [height, width, 3]): The origin of the camera coordinate.
        d (ndarray, [height, width, 3]): The direction of each ray.

    """
    _o = np.zeros((height, width, 4), dtype=np.float32)
    _o[:, :, 3] = 1

    v, u = np.mgrid[:height, :width].astype(np.float32)
    _x = (u - cx) / f
    _y = (v - cy) / f
    _z = np.ones_like(_x)
    _w = np.ones_like(_x)
    _d = np.stack([_x, _y, _z, _w], axis=2)

    o = (pose @ _o[..., None])[..., :3, 0]
    _d = (pose @ _d[..., None])[..., :3, 0]
    d = _d - o
    d /= np.linalg.norm(d, axis=2, keepdims=True)
    return o, d


class NeRF(nn.Module):

    # sampling parameter
    N_c = 64
    N_f = 128

    BATCH_SIZE = 2048

    def __init__(self, t_n=0., t_f=2.5, L_x=10, L_d=4, c_bg=(1, 1, 1)):
        self.t_n = t_n
        self.t_f = t_f
        self.c_bg = c_bg

        super(NeRF, self).__init__()
        self.rf_c = RadianceField(L_x=L_x, L_d=L_d)
        self.rf_f = RadianceField(L_x=L_x, L_d=L_d)

    def device(self):
        return next(self.parameters()).device

    def forward(self, view):
        """Render Image with view parameters.

        Args:
            view (dict): View (camera) parameters.
                view = {
                    # intrinsic parameters.
                    f: <float, the focal length.>,
                    cx : <float, the center of the image (x).>,
                    cy : <float, the center of the image (y).>,
                    width: <int, the image width.>,
                    height: <int, the image height.>,
                    # extrinsic parameter.
                    pose: <ndarray, [4, 4], camera extrinsic matrix.>
                }

        Returns:
            C_c (ndarray, [height, width, 3]): The rendered image (coarse).
            C_f (ndarray, [height, width, 3]): The rendered image (fine).

        """
        f = view['f']
        cx = view['cx']
        cy = view['cy']
        pose = view['pose']
        width = view['width']
        height = view['height']

        o, d = camera_params_to_rays(
            f, cx, cy, pose, width, height)
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)

        device = self.device()
        o = torch.tensor(o, device=device)
        d = torch.tensor(d, device=device)

        _C_c = []
        _C_f = []
        with torch.no_grad():
            for i in range(0, o.shape[0], self.BATCH_SIZE):
                o_i = o[i:i + self.BATCH_SIZE]
                d_i = d[i:i + self.BATCH_SIZE]
                C_c_i, C_f_i = self.infer(o_i, d_i)
                _C_c.append(C_c_i.cpu().numpy())
                _C_f.append(C_f_i.cpu().numpy())

        C_c = np.concatenate(_C_c, axis=0)
        C_f = np.concatenate(_C_f, axis=0)
        C_c = np.clip(0., 1., C_c.reshape(height, width, 3))
        C_f = np.clip(0., 1., C_f.reshape(height, width, 3))

        return C_c, C_f

    def infer(self, o, d):
        device = self.device()
        o = torch.tensor(o, device=device)
        d = torch.tensor(d, device=device)
        C_c, C_f = self.volume_rendering_with_radiance_field(o, d)
        return C_c, C_f

    def volume_rendering_with_radiance_field(self, o, d):
        """Rendering with Neural Radiance Field.

        Args:
            o (ndarray, [batch_size, 3]): Start points of the ray.
            d (ndarray, [batch_size, 3]): Directions of the ray.

        Returns:
            C_c (tensor, [batch_size, 3]): Result of coarse rendering.
            C_f (tensor, [batch_size, 3]): Result of fine rendering.

        """
        batch_size = o.shape[0]
        device = o.device

        partitions = split_ray(self.t_n, self.t_f, self.N_c, batch_size)

        # background.
        bg = torch.tensor(self.c_bg, device=device, dtype=torch.float32)
        bg = bg.view(1, 3)

        # coarse rendering:
        _t_c = sample_coarse(partitions)
        t_c = torch.tensor(_t_c)
        t_c = t_c.to(device)

        rgb_c, w_c = rgb_and_weight(self.rf_c, o, d, t_c, self.N_c)
        C_c = torch.sum(w_c[..., None] * rgb_c, axis=1)
        C_c += (1. - torch.sum(w_c, axis=1, keepdims=True)) * bg

        # fine rendering.
        _w_c = w_c.detach().cpu().numpy()
        t_f = sample_fine(partitions, _w_c, _t_c, self.N_f)
        t_f = torch.tensor(t_f)
        t_f = t_f.to(device)

        rgb_f, w_f = rgb_and_weight(self.rf_f, o, d, t_f, self.N_f + self.N_c)
        C_f = torch.sum(w_f[..., None] * rgb_f, axis=1)
        C_f += (1. - torch.sum(w_f, axis=1, keepdims=True)) * bg

        return C_c, C_f
