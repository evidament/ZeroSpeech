import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import json

from dataset import SpeechDataset, collate_fn
from model import Model

from tqdm import tqdm

with open("config.json") as file:
    params = json.load(file)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model(in_channels=params["preprocessing"]["num_mels"],
              encoder_channels=params["model"]["encoder"]["channels"],
              num_codebook_embeddings=params["model"]["codebook"]["num_embeddings"],
              codebook_embedding_dim=params["model"]["codebook"]["embedding_dim"],
              num_speakers=params["model"]["vocoder"]["num_speakers"],
              speaker_embedding_dim=params["model"]["vocoder"]["speaker_embedding_dim"],
              conditioning_channels=params["model"]["vocoder"]["conditioning_channels"],
              embedding_dim=params["model"]["vocoder"]["embedding_dim"],
              rnn_channels=params["model"]["vocoder"]["rnn_channels"],
              fc_channels=params["model"]["vocoder"]["fc_channels"],
              bits=params["preprocessing"]["bits"],
              hop_length=params["preprocessing"]["hop_length"],
              jitter=params["model"]["codebook"]["jitter"])
model.to(device)

# print("Load checkpoint from: {}:".format(args.checkpoint))
checkpoint = torch.load("epx/model.ckpt-400000.pt", map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint["model"])
model.eval()

dataset = SpeechDataset(root="datasetsv2/2019/english",
                        hop_length=params["preprocessing"]["hop_length"],
                        sample_rate=params["preprocessing"]["sample_rate"],)

dataloader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=8, collate_fn=collate_fn,
                        pin_memory=True, drop_last=True)

mlp = nn.Sequential(
    nn.Linear(64, 2048),
    nn.ReLU(True),
    nn.Linear(2048, 102)
)
mlp.to(device)

optimizer = optim.Adam(mlp.parameters(), lr=4e-4)

for epoch in range(100):
    average_loss = average_accuracy = 0
    for i, (mels, speakers, lengths) in enumerate(tqdm(dataloader), 1):
        mels, speakers, lengths = mels.to(device), speakers.to(device), lengths.to(device)
        with torch.no_grad():
            z = model.encode(mels.transpose(1, 2))
        downsampled_lengths = lengths // 2 - 1
        z = pack_padded_sequence(z, downsampled_lengths, batch_first=True, enforce_sorted=False)
        z, _ = pad_packed_sequence(z, batch_first=True)
        z = torch.sum(z, dim=1) / downsampled_lengths.float().unsqueeze(-1)
        logits = mlp(z)
        # logits = torch.mean(outputs, dim=1)
        loss = F.cross_entropy(logits, speakers)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = logits.argmax(dim=1) == speakers
        accuracy = torch.mean(accuracy.float())

        average_loss += (loss.item() - average_loss) / i
        average_accuracy += (accuracy.item() - average_accuracy) / i

    print("epoch {}, loss {}, accuracy {}".format(epoch, average_loss, average_accuracy))
