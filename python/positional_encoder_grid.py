""" Multi Resolution Hash Encoder
"""
import math
import torch
from torch import nn
from typing import Tuple


class PositionalEncoderGrid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l = 16
        self.t = 2 ** 14
        self.f = 2
        n_min = 16
        n_max = 512

        b = math.exp((math.log(n_max) - math.log(n_min)) / (self.l - 1))
        self.ns = [int(n_min * (b ** i)) for i in range(self.l)]

        # Prime Numbers from
        # https://github.com/NVlabs/tiny-cuda-nn/blob/ee585fa47e99de4c26f6ae88be7bcb82b9295310/include/tiny-cuda-nn/encodings/grid.h#L83
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861]))

        self.hash_table = nn.Parameter(
            torch.rand([self.l, self.t, self.f], requires_grad=True) * 2e-4 - 1e-4)

        self.bound = 3

    def encoded_dim(self) -> int:
        return self.l * self.f

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode positions by multi resolution hash.

        Args:
            inputs (torch.Tensor, [batch_size, dim]): Position.
            bound (float): range.

        Returns:
            torch.Tensor [batch_size, self.f * self.l]: Encoded position.

        """
        inputs = (inputs + self.bound) / (2 * self.bound)  # map to [0, 1]
        assert (inputs >= 0).all() and (inputs <= 1).all(), f'inputs must be in [0, 1]. min = {inputs.min()}, max = {inputs.max()}'

        feature_list = list()

        def get_min_max_per(value: torch.Tensor, width: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            min = (value // width).long()
            max = min + 1
            per = (value - min * width) / width
            return min, max, per

        for l, n in enumerate(self.ns):
            width = 1 / n
            x_min, x_max, x_per = get_min_max_per(inputs[:, 0], width)
            y_min, y_max, y_per = get_min_max_per(inputs[:, 1], width)
            z_min, z_max, z_per = get_min_max_per(inputs[:, 2], width)

            # (min, min, min)
            i1 = ((x_min * self.primes[0]) ^ (y_min * self.primes[1]) ^ (z_min * self.primes[2])) % self.t
            f1 = self.hash_table[l, i1]
            c1 = ((1 - x_per) * (1 - y_per) * (1 - z_per)).unsqueeze(-1)

            # (min, min, max)
            i2 = ((x_min * self.primes[0]) ^ (y_min * self.primes[1]) ^ (z_max * self.primes[2])) % self.t
            f2 = self.hash_table[l, i2]
            c2 = ((1 - x_per) * (1 - y_per) * z_per).unsqueeze(-1)

            # (min, max, min)
            i3 = ((x_min * self.primes[0]) ^ (y_max * self.primes[1]) ^ (z_min * self.primes[2])) % self.t
            f3 = self.hash_table[l, i3]
            c3 = ((1 - x_per) * y_per * (1 - z_per)).unsqueeze(-1)

            # (min, max, max)
            i4 = ((x_min * self.primes[0]) ^ (y_max * self.primes[1]) ^ (z_max * self.primes[2])) % self.t
            f4 = self.hash_table[l, i4]
            c4 = ((1 - x_per) * y_per * z_per).unsqueeze(-1)

            # (max, min, min)
            i5 = ((x_max * self.primes[0]) ^ (y_min * self.primes[1]) ^ (z_min * self.primes[2])) % self.t
            f5 = self.hash_table[l, i5]
            c5 = (x_per * (1 - y_per) * (1 - z_per)).unsqueeze(-1)

            # (max, min, max)
            i6 = ((x_max * self.primes[0]) ^ (y_min * self.primes[1]) ^ (z_max * self.primes[2])) % self.t
            f6 = self.hash_table[l, i6]
            c6 = (x_per * (1 - y_per) * z_per).unsqueeze(-1)

            # (max, max, min)
            i7 = ((x_max * self.primes[0]) ^ (y_max * self.primes[1]) ^ (z_min * self.primes[2])) % self.t
            f7 = self.hash_table[l, i7]
            c7 = (x_per * y_per * (1 - z_per)).unsqueeze(-1)

            # (max, max, max)
            i8 = ((x_max * self.primes[0]) ^ (y_max * self.primes[1]) ^ (z_max * self.primes[2])) % self.t
            f8 = self.hash_table[l, i8]
            c8 = (x_per * y_per * z_per).unsqueeze(-1)

            feature = f1 * c1 + f2 * c2 + f3 * c3 + f4 * c4 + f5 * c5 + f6 * c6 + f7 * c7 + f8 * c8
            feature_list.append(feature)

        result = torch.cat(feature_list, dim=1)
        return result


if __name__ == '__main__':
    encoder = PositionalEncoderGrid()
    x = torch.rand((256, 3))
    print(f"x.shape = {x.shape}")
    feature_map = encoder(x)
    print(f"feature_map.shape = {feature_map.shape}")
