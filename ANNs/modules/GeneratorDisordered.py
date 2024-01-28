import torch
import torch.nn as nn


class GeneratorDisordered(nn.Module):
    def __init__(self, challenge_bits, ns, act):
        super().__init__()
        self.challenge_bits = challenge_bits
        if act == "LeakyReLU":
            slope = 0.2
            activation = nn.LeakyReLU(slope, inplace=True)
        elif act == "GELU":
            activation = nn.GELU()
        elif act == "ELU":
            activation = nn.ELU()

        self.main = nn.Sequential(
            nn.ConvTranspose1d(challenge_bits, ns * 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 64),
            activation,

            nn.ConvTranspose1d(ns * 64, ns * 32, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 32),
            activation,

            nn.ConvTranspose1d(ns * 32, ns *16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 16),
            activation,

            nn.ConvTranspose1d(ns * 16, ns * 8, 8, 4, 2, bias=False),
            nn.BatchNorm1d(ns * 8),
            activation,

            nn.ConvTranspose1d(ns * 8, ns * 4, 8, 4, 2, bias=False),
            nn.BatchNorm1d(ns * 4),
            activation,

            nn.ConvTranspose1d(ns * 4, ns * 2, 16, 4, 0, bias=False),
            nn.BatchNorm1d(ns * 2),
            activation,

            nn.ConvTranspose1d(ns * 2, ns, 16, 1, 2, bias=False),
            nn.BatchNorm1d(ns),
            activation,

            nn.ConvTranspose1d(ns, 1, 17, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, self.challenge_bits, 1)
        return torch.squeeze(self.main(input))
