import numpy as np
import torch
from torch.utils.data import Dataset
import random
import json


class MelDataset(Dataset):
    def __init__(self, metadata_paths, sample_frames):
        self.sample_frames = sample_frames

        metadata = dict()
        for path in metadata_paths:
            with path.open() as file:
                metadata.update(json.load(file))

        self.speakers = sorted(metadata.keys())

        self.metadata = list()
        for speaker, paths in metadata.items():
            self.metadata.extend([(speaker, path) for path in paths])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        speaker_id, path = self.metadata[index]
        mel = np.load(path)

        pos = random.randint(0, mel.shape[0] - self.sample_frames)
        mel = mel[pos:pos + self.sample_frames, :]

        speaker = self.speakers.index(speaker_id)

        return torch.FloatTensor(mel), speaker
