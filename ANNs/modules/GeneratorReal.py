import torch.nn as nn


class GeneratorReal(nn.Module):
    def __init__(self, ns, challenge_bits, c_weight):
        super().__init__()
        self.challenge_bits = challenge_bits
        self.c_weight = c_weight
        self.init_dim = 64
        self.challenge = nn.Linear(challenge_bits, self.init_dim ** 2 * c_weight)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(c_weight, ns * 8, 3, 2, 1, output_padding=(1,), bias=False),
            nn.BatchNorm2d(ns * 8),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 8, ns * 4, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 4, ns * 2, 3, 2, 1, output_padding=(1,), bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns),
            nn.GELU(),

            nn.ConvTranspose2d(ns, 1, 3, 2, 1, output_padding=(1,), bias=False),
            nn.Tanh()
        )

        '''self.main = nn.Sequential(
            nn.ConvTranspose2d(c_weight, ns * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 8),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 8, ns * 4, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 4, ns * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns),
            nn.GELU(),

            nn.ConvTranspose2d(ns, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )'''

    def forward(self, challenge_input):
        challenge_input = challenge_input.view(-1, self.challenge_bits)
        challenge_input = self.challenge(challenge_input)
        challenge_input = challenge_input.view(-1, self.c_weight, self.init_dim, self.init_dim)

        return self.main(challenge_input)
