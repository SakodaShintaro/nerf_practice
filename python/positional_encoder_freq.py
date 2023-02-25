import torch
from torch import nn


class PositionalEncoderFreq(nn.Module):
    def __init__(self, L: int) -> None:
        super().__init__()
        self.L = L

    def encoded_dim(self):
        return 6 * self.L

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Encode positions by sin, cos

        Args:
            p (torch.Tensor, [batch_size, dim]): Position.

        Returns:
            torch.Tensor [batch_size, dim * 2 * L]: Encoded position.

        """
        # normalization.
        p = torch.tanh(p)

        batch_size = p.shape[0]
        i = torch.arange(self.L, dtype=torch.float32, device=p.device)
        a = (2. ** i[None, None]) * torch.pi * p[:, :, None]
        s = torch.sin(a)
        c = torch.cos(a)
        e = torch.cat([s, c], dim=2).view(batch_size, -1)
        return e
