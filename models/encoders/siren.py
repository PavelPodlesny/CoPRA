import torch
import torch.nn as nn


class Siren(nn.Module):
    def __init__(self, dim):
        super().__init__()
        freq = torch.ones(dim, dtype=torch.float32) + torch.normal(0, .1, (dim,))
        self.freq = torch.nn.parameter.Parameter(freq, requires_grad=True)

        p = torch.zeros(dim, dtype=torch.float32)
        p[1::2] = torch.pi/2
        self.phase = torch.nn.parameter.Parameter(p, requires_grad=True)

    def forward(self, x):
        x = x*torch.pi
        x = x[..., None]*self.freq + self.phase
        x = torch.sin(x)

        return x
