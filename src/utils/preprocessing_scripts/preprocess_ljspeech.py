import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split

SEED = 0xDEADF00D


def preprocess_ljspeech(root_path: Path, num_val_samples: int = 100, tgt_wav_folder: str = "speaker_0"):
    """Extracting only normalized text"""
    meta_path = root_path / "metadata.csv"
    df = pd.read_csv(meta_path, sep="|", header=None)
    df.dropna(inplace=True)
    train_df, dev_df = train_test_split(df, test_size=num_val_samples, random_state=SEED)

    (root_path / "wavs" / tgt_wav_folder).mkdir(exist_ok=True)

    with open(root_path / "train.json", "w") as train_file:
        for _, row in train_df.iterrows():
            src_wav_path = root_path / "wavs" / f"{row[0]}.wav"
            tgt_wav_path = root_path / "wavs" / tgt_wav_folder / f"{row[0]}.wav"
            shutil.move(src_wav_path, tgt_wav_path)

            wav, sr = torchaudio.load(tgt_wav_path)
            duration = wav.shape[1] / sr

            d = {
                "audio_filepath": str(tgt_wav_path.resolve()),
                "text": row[2],
                "duration": duration,
                "speaker": 0,
            }
            train_file.write(json.dumps(d))
            train_file.write("\n")

    with open(root_path / "dev.json", "w") as dev_file:
        for _, row in dev_df.iterrows():
            src_wav_path = root_path / "wavs" / f"{row[0]}.wav"
            tgt_wav_path = root_path / "wavs" / tgt_wav_folder / f"{row[0]}.wav"
            shutil.move(src_wav_path, tgt_wav_path)

            wav, sr = torchaudio.load(tgt_wav_path)
            duration = wav.shape[1] / sr

            d = {
                "audio_filepath": str(tgt_wav_path.resolve()),
                "text": row[2],
                "duration": duration,
                "speaker": 0,
            }
            dev_file.write(json.dumps(d))
            dev_file.write("\n")

    for p in (root_path / "wavs").iterdir():
        if not p.is_dir():
            p.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    args = parser.parse_args()

    preprocess_ljspeech(args.root_path)
