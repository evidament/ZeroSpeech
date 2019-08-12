import argparse
import numpy as np
import json
from pathlib import Path
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from utils import load_wav, melspectrogram


def process_wav(wav_path, mel_out_dir, params):
    speaker = wav_path.stem.split("_")[0]
    mel_path = mel_out_dir / wav_path.stem
    mel_path = mel_path.with_suffix(".npy")

    wav = load_wav(str(wav_path), sample_rate=params["preprocessing"]["sample_rate"])
    if wav.shape[0] < params["preprocessing"]["sample_frames"] * params["preprocessing"]["hop_length"]:
        return None

    wav /= np.abs(wav).max() * 0.999

    mel = melspectrogram(wav, sample_rate=params["preprocessing"]["sample_rate"],
                         preemph=params["preprocessing"]["preemph"],
                         num_mels=params["preprocessing"]["num_mels"],
                         num_fft=params["preprocessing"]["num_fft"],
                         min_level_db=params["preprocessing"]["min_level_db"],
                         hop_length=params["preprocessing"]["hop_length"],
                         win_length=params["preprocessing"]["win_length"],
                         fmin=params["preprocessing"]["fmin"])

    np.save(mel_path, mel)
    return wav_path.parts[-2], speaker, mel_path, mel.shape[0]


def preprocess(data_dir, out_dir, num_workers, params):
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    for wav_path in data_dir.rglob("*.wav"):
        mel_out_dir = out_dir.joinpath(wav_path.parts[-2])
        mel_out_dir.mkdir(parents=True, exist_ok=True)
        futures.append(executor.submit(partial(process_wav, wav_path, mel_out_dir, params)))

    results = [future.result() for future in tqdm(futures)]
    write_metadata(results, out_dir, params)


def write_metadata(results, out_dir, params):
    metadata = dict()
    for split, speaker, path, _ in filter(lambda x: x is not None, results):
        metadata.setdefault(split, {}).setdefault(speaker, []).append(str(path))

    for split, content in metadata.items():
        split_path = out_dir / "{}.json".format(split)
        with split_path.open("w") as file:
            json.dump(content, file)

    lengths = [x[-1] for x in filter(lambda x: x is not None, results)]
    frames = sum(lengths)
    frame_shift_ms = params["preprocessing"]["hop_length"] / params["preprocessing"]["sample_rate"]
    hours = frames * frame_shift_ms / 3600
    print("Wrote {} utterances, {} frames ({:.2f} hours)".format(len(lengths), frames, hours))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./ZeroSpeech2019/english")
    parser.add_argument("--out-dir", default="./data")
    parser.add_argument("--num-workers", type=int, default=cpu_count())
    with Path("config.json").open() as f:
        params = json.load(f)
    args = parser.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    preprocess(data_dir, out_dir, args.num_workers, params)


if __name__ == "__main__":
    main()
