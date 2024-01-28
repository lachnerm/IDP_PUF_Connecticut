import torch
import torch.nn as nn

class Generator8_complex(nn.Module):
    def __init__(self, challenge_bits, ns):
        super().__init__()
        self.challenge_bits = challenge_bits

        slope = 0.2
        self.main = nn.Sequential(
            nn.ConvTranspose1d(challenge_bits, ns * 16, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 16),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose1d(ns * 16, ns * 8, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 8),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose1d(ns * 8, ns * 4, 4, 2, 1, bias=False),
            nn.BatchNorm1d(ns * 4),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose1d(ns * 4, ns * 2, 8, 4, 2, bias=False),
            nn.BatchNorm1d(ns * 2),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose1d(ns * 2, ns, 8, 4, 2, bias=False),
            nn.BatchNorm1d(ns),
            nn.LeakyReLU(slope, inplace=True),

            nn.ConvTranspose1d(ns, 2, 13, 4, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        input = input.view(-1, self.challenge_bits, 1)
        output = torch.squeeze(self.main(input))
        #return output.view(-1, 521, 2)
        return output