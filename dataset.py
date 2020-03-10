import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sample_rate, sample_frames=None):
        self.root = Path(root)
        self.sample_frames = sample_frames
        self.hop_length = hop_length

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sample_rate if sample_frames is not None else 0
        with open(self.root / "train.json") as file:
            metadata = json.load(file)
            self.metadata = [
                Path(out_path) for _, _, duration, out_path in metadata
                if float(duration) > min_duration
            ]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root.parent / path

        audio = np.load(path.with_suffix(".wav.npy"))
        mel = np.load(path.with_suffix(".mel.npy"))

        if self.sample_frames is not None:
            pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
            mel = mel[:, pos - 1:pos + self.sample_frames + 1]
            audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(path.parts[-2])

        return torch.from_numpy(audio), torch.from_numpy(mel), speaker


def collate_fn(batch):
    mels = pad_sequence([item[1].transpose(0, 1) for item in batch], batch_first=True)
    lengths = torch.LongTensor([item[1].shape[-1] for item in batch])
    speakers = torch.LongTensor([item[-1] for item in batch])
    return mels, speakers, lengths
