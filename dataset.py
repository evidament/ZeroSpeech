import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
from random import randint
from pathlib import Path


class SpeechDataset(Dataset):
    def __init__(self, root, hop_length, sample_rate, sample_frames):
        self.root = Path(root)
        self.hop_length = hop_length
        self.sample_frames = sample_frames

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        min_duration = (sample_frames + 2) * hop_length / sample_rate
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

        pos = randint(1, mel.shape[-1] - self.sample_frames - 2)
        mel = mel[:, pos - 1:pos + self.sample_frames + 1]
        audio = audio[pos * self.hop_length:(pos + self.sample_frames) * self.hop_length + 1]

        speaker = self.speakers.index(path.parts[-2])

        return torch.LongTensor(audio), torch.FloatTensor(mel), speaker


class ProbeDataset(Dataset):
    def __init__(self, root, hop_length, sample_rate):
        self.root = Path(root)
        self.hop_length_ms = hop_length / sample_rate

        with open(self.root / "speakers.json") as file:
            self.speakers = sorted(json.load(file))

        with open(self.root / "train.json") as file:
            metadata = json.load(file)
            self.metadata = [
                Path(out_path) for _, _, _, out_path in metadata
            ]

        with open(self.root / "phones.json") as file:
            self.phones = sorted(json.load(file))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        path = self.metadata[index]
        path = self.root.parent / path

        mel = np.load(path.with_suffix(".mel.npy"))
        with open(path.with_suffix(".align.json")) as file:
            alignments = json.load(file)

        phones = [self.phones.index(phone) for _, _, phone in alignments]
        lengths = [round((end - start)/self.hop_length_ms) for start, end, _ in alignments]
        phones = np.repeat(phones, lengths)

        speaker = self.speakers.index(path.parts[-2])

        return torch.FloatTensor(mel), torch.LongTensor(phones), speaker


def collate_fn(batch):
    mels = pad_sequence([item[0].transpose(0, 1) for item in batch], batch_first=True)
    lengths = torch.LongTensor([item[1].shape[-1] for item in batch])
    speakers = torch.LongTensor([item[-1] for item in batch])
    return mels, speakers, lengths
