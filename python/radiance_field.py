import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from positional_encoder_freq import PositionalEncoderFreq


class SkipConnection(nn.Module):
    def __init__(self, ch_num: int, layer_num: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(ch_num, ch_num) for _ in range(layer_num)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            y = x
            x = layer(x)
            x = F.relu(x)
            x = x + y
        return x


class RadianceField(nn.Module):
    """Radiance Field Functions.

    This is ``$F_\\Theta$`` in the paper.

    """

    def __init__(self, L_x: int = 10, L_d: int = 4) -> None:
        super(RadianceField, self).__init__()

        # positional encoding parameter.
        self.enc_pos = PositionalEncoderFreq(L_x)
        self.enc_dir = PositionalEncoderFreq(L_d)

        self.layer0 = nn.Linear(self.enc_pos.encoded_dim(), 256)
        self.skip0 = SkipConnection(256, 4)
        self.layer5 = nn.Linear(256 + self.enc_pos.encoded_dim(), 256)
        self.skip1 = SkipConnection(256, 3)
        self.sigma = nn.Linear(256, 1)
        self.layer9 = nn.Linear(256 + self.enc_dir.encoded_dim(), 128)
        self.skip2 = SkipConnection(128, 3)
        self.rgb = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor, d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply function.

        Args:
            x (tensor, [batch_size, 3]): Points on rays.
            d (tensor, [batch_size, 3]): Direction of rays.

        Returns:
            rgb (tensor, [batch_size, 3]): Emitted color.
            sigma (tensor, [batch_size, 1]): Volume density.

        """
        # positional encoding.
        e_x = self.enc_pos(x)
        e_d = self.enc_dir(d)

        # forward
        h = F.relu(self.layer0(e_x))
        h = self.skip0(h)
        h = torch.cat([h, e_x], dim=1)
        h = F.relu(self.layer5(h))
        h = self.skip1(h)
        sigma = F.relu(self.sigma(h))
        h = torch.cat([h, e_d], dim=1)
        h = F.relu(self.layer9(h))
        h = self.skip2(h)
        rgb = torch.sigmoid(self.rgb(h))

        return rgb, sigma
