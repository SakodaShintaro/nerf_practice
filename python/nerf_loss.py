import torch
import torch.nn as nn
import torch.nn.functional as F
from sample_function import volume_rendering_with_radiance_field


class NeRFLoss(nn.Module):

    def __init__(self, nerf):
        super(NeRFLoss, self).__init__()
        self.nerf = nerf

    def forward(self, o, d, C):
        device = self.nerf.device()
        o = torch.tensor(o, device=device)
        d = torch.tensor(d, device=device)
        C = torch.tensor(C, device=device)

        rf_c = self.nerf.rf_c
        rf_f = self.nerf.rf_f
        t_n = self.nerf.t_n
        t_f = self.nerf.t_f
        N_c = self.nerf.N_c
        N_f = self.nerf.N_f
        c_bg = self.nerf.c_bg
        C_c, C_f = volume_rendering_with_radiance_field(
            rf_c, rf_f, o, d, t_n, t_f, N_c=N_c, N_f=N_f, c_bg=c_bg)

        loss = F.mse_loss(C_c, C) + F.mse_loss(C_f, C)
        return loss
