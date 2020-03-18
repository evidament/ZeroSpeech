import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from model import Model


def encode_dataset(args, params):
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

    print("Load checkpoint from: {}:".format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    in_dir = Path(args.in_dir)
    for path in tqdm(in_dir.rglob("*.mel.npy")):
        mel = torch.from_numpy(np.load(path)).unsqueeze(0).to(device)
        with torch.no_grad():
            z, indices = model.encode(mel)

        z = z.squeeze().cpu().numpy()
        relative_path = path.relative_to(in_dir)
        relative_path = relative_path.with_suffix("")
        out_path = out_dir / relative_path
        out_path.parent.mkdir(exist_ok=True, parents=True)

        with open(out_path.with_suffix(".txt"), "w") as file:
            np.savetxt(file, z)

        if args.save_mean:
            mean = np.mean(z, 0)
            with open(out_path.with_suffix(".mean.txt"), "w") as file:
                np.savetxt(file, mean)

        if args.save_indices:
            indices = indices.squeeze().cpu().numpy()
            with open(out_path.with_suffix(".index.txt"), "w") as file:
                np.savetxt(file, indices, fmt="%i")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, help="Checkpoint path to resume")
    parser.add_argument("--in-dir", type=str, help="Directory to encode")
    parser.add_argument("--out-dir", type=str, help="Output path")
    parser.add_argument("--save-mean", action="store_true")
    parser.add_argument("--save-indices", action="store_true")
    args = parser.parse_args()
    with open("config.json") as file:
        params = json.load(file)
    encode_dataset(args, params)