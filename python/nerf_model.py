import numpy as np
import torch
import torch.nn as nn
from radiance_field import RadianceField
from sample_function import volume_rendering_with_radiance_field


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

    # batchsize
    N_SAMPLES = 2048

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
        """Render Image with view paramters.

        Args:
            view (dict): View (camera) parameters.
                view = {
                    # intrinsic paramters.
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
            for i in range(0, o.shape[0], self.N_SAMPLES):
                o_i = o[i:i + self.N_SAMPLES]
                d_i = d[i:i + self.N_SAMPLES]
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

        rf_c = self.rf_c
        rf_f = self.rf_f
        t_n = self.t_n
        t_f = self.t_f
        N_c = self.N_c
        N_f = self.N_f
        c_bg = self.c_bg
        C_c, C_f = volume_rendering_with_radiance_field(
            rf_c, rf_f, o, d, t_n, t_f, N_c=N_c, N_f=N_f, c_bg=c_bg)
        return C_c, C_f
