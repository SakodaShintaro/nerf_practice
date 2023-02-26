""" Multi Resolution Hash Encoder
"""
import math
import torch
from torch import nn


class PositionalEncoderGrid(nn.Module):
    def __init__(self, l=16, t=2**14, f=2, n_min=16, n_max=512):
        super().__init__()
        self.l = l
        self.t = t
        self.f = f

        b = math.exp((math.log(n_max) - math.log(n_min)) / (l - 1))
        self.ns = [int(n_min * (b ** i)) for i in range(l)]

        # Prime Numbers from
        # https://github.com/NVlabs/tiny-cuda-nn/blob/ee585fa47e99de4c26f6ae88be7bcb82b9295310/include/tiny-cuda-nn/encodings/grid.h#L83
        self.register_buffer('primes', torch.tensor([1, 2654435761, 805459861]))

        self.hash_table = nn.Parameter(
            torch.rand([l, t, f], requires_grad=True) * 2e-4 - 1e-4)
        
        self.bound = 3.0

    def encoded_dim(self):
        return self.l * self.f

    def forward(self, inputs: torch.Tensor):
        """Encode positions by multi resolution hash.

        Args:
            inputs (torch.Tensor, [batch_size, dim]): Position.
            bound (float): range.

        Returns:
            torch.Tensor [batch_size, self.f * self.l]: Encoded position.

        """
        inputs = (inputs + self.bound) / (2 * self.bound)  # map to [0, 1]
        assert (inputs >= 0).all() and (inputs <= 1).all(), 'inputs must be in [0, 1]'

        feature_list = list()

        for l, n in enumerate(self.ns):
            width = 1 / n
            x_min = (inputs[:, 0] // width).long()
            x_max = x_min + 1
            x_per = (inputs[:, 0] - x_min * width) / width

            y_min = (inputs[:, 1] // width).long()
            y_max = y_min + 1
            y_per = (inputs[:, 1] - y_min * width) / width

            z_min = (inputs[:, 2] // width).long()
            z_max = z_min + 1
            z_per = (inputs[:, 2] - z_min * width) / width

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
