import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 64 * 2, 1)
        )

    def forward(self, mels):
        x = self.conv(mels.transpose(1, 2))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(102, 256)
        self.conv = nn.Sequential(
            nn.Conv1d(64 + 256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.ConvTranspose1d(512, 512, 4, 2, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 3, 1, 1, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 80, 1)
        )

    def forward(self, z, speakers):
        speakers = self.embedding(speakers)
        speakers = speakers.unsqueeze(-1).expand(-1, -1, z.size(-1))

        z = torch.cat((z, speakers), dim=1)
        mels = self.conv(z)
        mels = mels.transpose(1, 2)
        return mels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, mels, speakers):
        mu, logvar = self.encoder(mels).split(64, dim=1)
        z = reparameterize(mu, logvar)
        mels = self.decoder(z, speakers)
        return mels, mu, logvar
