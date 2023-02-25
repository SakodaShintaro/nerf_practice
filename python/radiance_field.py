import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from positional_encoder_freq import PositionalEncoderFreq

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
        self.layer1 = nn.Linear(256, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, 256)
        self.layer4 = nn.Linear(256, 256)
        self.layer5 = nn.Linear(256 + self.enc_pos.encoded_dim(), 256)
        self.layer6 = nn.Linear(256, 256)
        self.layer7 = nn.Linear(256, 256)
        self.sigma = nn.Linear(256, 1)
        self.layer8 = nn.Linear(256, 256)
        self.layer9 = nn.Linear(256 + self.enc_dir.encoded_dim(), 128)
        self.layer10 = nn.Linear(128, 128)
        self.layer11 = nn.Linear(128, 128)
        self.layer12 = nn.Linear(128, 128)
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
        h = F.relu(self.layer1(h))
        h = F.relu(self.layer2(h))
        h = F.relu(self.layer3(h))
        h = F.relu(self.layer4(h))
        h = torch.cat([h, e_x], dim=1)
        h = F.relu(self.layer5(h))
        h = F.relu(self.layer6(h))
        h = F.relu(self.layer7(h))
        sigma = F.relu(self.sigma(h))
        h = self.layer8(h)
        h = torch.cat([h, e_d], dim=1)
        h = F.relu(self.layer9(h))
        h = F.relu(self.layer10(h))
        h = F.relu(self.layer11(h))
        h = F.relu(self.layer12(h))
        rgb = torch.sigmoid(self.rgb(h))

        return rgb, sigma

