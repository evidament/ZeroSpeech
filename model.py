import torch
import torch.nn as nn
import torch.nn.functional as F


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1. / self.num_embeddings, 1. / self.num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, x):
        x_flatten = x.view(-1, self.embedding_dim)

        # Compute the distances to the codebook
        distances = torch.addmm(torch.sum(self.embedding.weight ** 2, dim=1) +
                                torch.sum(x_flatten ** 2, dim=1, keepdim=True),
                                x_flatten, self.embedding.weight.t(),
                                alpha=-2.0, beta=1.0)

        _, indices = torch.min(distances, dim=1)
        quantized = self.embedding(indices)
        quantized = quantized.view_as(x)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        encodings = torch.zeros(indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, indices.unsqueeze(1), 1)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(80, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.rnn = nn.LSTM(512, 32, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, mels):
        x = self.conv(mels.transpose(1, 2))
        x = x.transpose(1, 2)
        x, _ = self.rnn(x)
        f, b = x.split(32, dim=-1)
        f, b = f[:, 4-1::4, :], b[:, 0:-(4-1):4, :]
        x = torch.cat((f, b), dim=-1)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(102, 256)
        self.rnn1 = nn.LSTM(64 + 256, 512, batch_first=True)
        self.conv = nn.Sequential(
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Conv1d(512, 512, 5, 1, 2, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True)
        )
        self.rnn2 = nn.LSTM(512, 1024, num_layers=2, batch_first=True)
        self.proj = nn.Linear(1024, 80)

    def forward(self, x, speakers):
        x = F.interpolate(x.transpose(1, 2), scale_factor=4)
        x = x.transpose(1, 2)

        speakers = self.embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, x.size(1), -1)

        x = torch.cat((x, speakers), dim=-1)
        x, _ = self.rnn1(x)

        x = self.conv(x.transpose(1, 2))

        x = x.transpose(1, 2)
        x, _ = self.rnn2(x)
        mels = self.proj(x)
        return mels


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.codebook = VQEmbedding(512, 64)
        self.decoder = Decoder()

    def forward(self, mels, speakers):
        x = self.encoder(mels)
        x, loss, perplexity = self.codebook(x)
        mels = self.decoder(x, speakers)
        return mels, loss, perplexity
