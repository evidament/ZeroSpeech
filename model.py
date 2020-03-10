import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from preprocess import mulaw_decode


class Jitter(nn.Module):
    def __init__(self, p):
        super().__init__()
        prob = torch.Tensor([p / 2, 1 - p, p / 2])
        self.register_buffer("prob", prob)

    def forward(self, x):
        if not self.training:
            return x
        else:
            batch_size, sample_size, channels = x.size()

            dist = Categorical(self.prob)
            index = dist.sample(torch.Size([batch_size, sample_size])) - 1
            index[:, 0].clamp_(0, 1)
            index[:, -1].clamp_(-1, 0)
            index += torch.arange(sample_size, device=x.device)

            x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
        return x


class VQEmbeddingEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        bound = 1 / num_embeddings
        embedding = torch.Tensor(num_embeddings, embedding_dim)
        embedding.uniform_(-bound, bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        M, D = self.embedding.size()

        x = x.transpose(1, 2)
        x_flat = x.detach().reshape(-1, D)

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, channels, 3, 1, 0, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 4, 2, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(True),
            nn.Conv1d(channels, out_channels, 1),
        )

    def forward(self, mels):
        return self.encoder(mels)


def get_gru_cell(gru):
    gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
    gru_cell.weight_hh.data = gru.weight_hh_l0.data
    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data
    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    return gru_cell


class Vocoder(nn.Module):
    def __init__(self, in_channels, num_speakers, speaker_embedding_dim, conditioning_channels, embedding_dim,
                 rnn_channels, fc_channels, bits, hop_length):
        super().__init__()
        self.rnn_channels = rnn_channels
        self.quantization_channels = 2**bits
        self.hop_length = hop_length

        self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
        self.rnn1 = nn.GRU(in_channels + speaker_embedding_dim, conditioning_channels,
                           num_layers=2, batch_first=True, bidirectional=True)
        self.embedding = nn.Embedding(self.quantization_channels, embedding_dim)
        self.rnn2 = nn.GRU(embedding_dim + 2 * conditioning_channels, rnn_channels, batch_first=True)
        self.fc1 = nn.Linear(rnn_channels, fc_channels)
        self.fc2 = nn.Linear(fc_channels, self.quantization_channels)
        
    def forward(self, x, mels, speakers):
        mels = F.interpolate(mels.transpose(1, 2), scale_factor=2)
        mels = mels.transpose(1, 2)

        speakers = self.speaker_embedding(speakers)
        speakers = speakers.unsqueeze(1).expand(-1, mels.size(1), -1)

        mels = torch.cat((mels, speakers), dim=-1)
        mels, _ = self.rnn1(mels)

        mels = F.interpolate(mels.transpose(1, 2), scale_factor=self.hop_length)
        mels = mels.transpose(1, 2)

        x = self.embedding(x)
        x, _ = self.rnn2(torch.cat((x, mels), dim=2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def generate(self, mel, speaker):
        output = []
        cell = get_gru_cell(self.rnn2)

        with torch.no_grad():
            mel = F.interpolate(mel.transpose(1, 2), scale_factor=2)
            mel = mel.transpose(1, 2)

            speaker = self.speaker_embedding(speaker)
            speaker = speaker.unsqueeze(1).expand(-1, mel.size(1), -1)

            mel = torch.cat((mel, speaker), dim=-1)
            mel, _ = self.rnn1(mel)

            mel = F.interpolate(mel.transpose(1, 2), scale_factor=self.hop_length)
            mel = mel.transpose(1, 2)

            batch_size, sample_size, _ = mel.size()

            h = torch.zeros(batch_size, self.rnn_channels, device=mel.device)
            x = torch.zeros(batch_size, device=mel.device).fill_(self.quantization_channels // 2).long()

            for m in tqdm(torch.unbind(mel, dim=1), leave=False):
                x = self.embedding(x)
                h = cell(torch.cat((x, m), dim=1), h)

                x = F.relu(self.fc1(h))
                logits = self.fc2(x)

                posterior = F.softmax(logits, dim=1)
                dist = Categorical(posterior)

                x = dist.sample()
                output.append(2 * x.float().item() / (self.quantization_channels - 1.) - 1.)

        output = np.asarray(output, dtype=np.float64)
        output = mulaw_decode(output, self.quantization_channels)
        return output


class Model(nn.Module):
    def __init__(self, in_channels, encoder_channels, num_codebook_embeddings, codebook_embedding_dim, num_speakers,
                 speaker_embedding_dim, conditioning_channels, embedding_dim, rnn_channels, fc_channels,
                 bits, hop_length, jitter=0):
        super(Model, self).__init__()
        self.encoder = Encoder(in_channels, encoder_channels, codebook_embedding_dim)
        self.codebook = VQEmbeddingEMA(num_codebook_embeddings, codebook_embedding_dim)
        self.decoder = Vocoder(codebook_embedding_dim, num_speakers, speaker_embedding_dim, conditioning_channels,
                               embedding_dim, rnn_channels, fc_channels, bits, hop_length)
        self.jitter = Jitter(jitter) if jitter > 0 else None

    def forward(self, x, mels, speakers):
        mels = self.encoder(mels)
        mels, loss, perplexity = self.codebook(mels)
        if self.jitter is not None:
            mels = self.jitter(mels)
        mels = self.decoder(x, mels, speakers)
        return mels, loss, perplexity

    def encode(self, mel):
        mel = self.encoder(mel)
        z, _, _ = self.codebook(mel)
        # z = mel.transpose(1, 2)
        return z

    def generate(self, mel, speaker):
        self.eval()
        with torch.no_grad():
            mel = self.encoder(mel)
            mel, _, perplexity = self.codebook(mel)
            output = self.decoder.generate(mel, speaker)
        self.train()
        return output
