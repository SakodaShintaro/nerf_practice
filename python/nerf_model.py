import numpy as np
import torch
import torch.nn as nn
from radiance_field import RadianceField
from nerf_function import split_ray, sample_coarse, sample_fine, rgb_and_weight
from typing import Tuple


class NeRF(nn.Module):

    # sampling parameter
    N_c = 64
    N_f = 128

    BATCH_SIZE = 2048

    def __init__(self, t_n: float = 0., t_f: float = 2.5, L_x: int = 10, L_d: int = 4) -> None:
        self.t_n = t_n
        self.t_f = t_f
        self.c_bg = (1, 1, 1)

        super(NeRF, self).__init__()
        self.rf_c = RadianceField(L_x=L_x, L_d=L_d)
        self.rf_f = RadianceField(L_x=L_x, L_d=L_d)

    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, o: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device()
        o = o.to(device)
        d = d.to(device)

        batch_size = o.shape[0]
        device = o.device

        partitions = split_ray(self.t_n, self.t_f, self.N_c, batch_size)
        partitions = partitions.to(device)
        partitions = partitions.detach()

        # background.
        bg = torch.tensor(self.c_bg, device=device, dtype=torch.float32)
        bg = bg.view(1, 3)

        # coarse rendering.
        _t_c = sample_coarse(partitions)
        t_c = _t_c.to(device)

        rgb_c, w_c = rgb_and_weight(self.rf_c, o, d, t_c, self.N_c)
        C_c = torch.sum(w_c[..., None] * rgb_c, dim=1)
        C_c += (1. - torch.sum(w_c, dim=1, keepdim=True)) * bg

        # fine rendering.
        _w_c = w_c.clone()
        t_f = sample_fine(partitions, _w_c, _t_c, self.N_f)
        t_f = t_f.to(device)

        rgb_f, w_f = rgb_and_weight(self.rf_f, o, d, t_f, self.N_f + self.N_c)
        C_f = torch.sum(w_f[..., None] * rgb_f, dim=1)
        C_f += (1. - torch.sum(w_f, dim=1, keepdim=True)) * bg

        return C_c, C_f
