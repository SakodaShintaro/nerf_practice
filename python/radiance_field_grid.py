import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from positional_encoder_grid import PositionalEncoderGrid
from positional_encoder_grid2 import HashEmbedder


class RadianceFieldGrid(nn.Module):
    """ Multi resolution hash encodingを使ったradiance field
    参考) https://github.com/yashbhalgat/HashNeRF-pytorch
    """

    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 ):
        super(RadianceFieldGrid, self).__init__()

        self.enc_pos = PositionalEncoderGrid()
        self.enc_dir = PositionalEncoderGrid()

        self.input_ch = self.enc_pos.encoded_dim()
        self.input_ch_views = self.enc_dir.encoded_dim()

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.enc_pos(x)
        d = self.enc_dir(d)

        # sigma
        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            h = F.relu(h, inplace=True)

        sigma, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        color = torch.sigmoid(h)

        return color, sigma
