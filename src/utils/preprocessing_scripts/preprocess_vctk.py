import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import torchaudio
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SEED = 0xDEADF00D


def preprocess_vctk(root_path: Path, pauses_csv: Optional[Path], num_val_samples: int = 100):
    txt_folder_names = [p.stem for p in (root_path / "txt").iterdir()]
    wav_folder_names = [p.stem for p in (root_path / "wav48").iterdir()]
    folder_names = list(set(txt_folder_names) & set(wav_folder_names))

    speaker_mapping = {}
    data = []

    if pauses_csv is not None:
        pauses_df = pd.read_csv(pauses_csv, header=None, sep="|")
        border_map = {}
        for _, row in pauses_df.iterrows():
            file_name = Path(row[0]).stem
            borders = row[3], row[4]
            border_map[file_name] = borders

    file_count, border_file_count = 0, 0
    for name in tqdm(folder_names):
        txt_folder, wav_folder = root_path / "txt" / name, root_path / "wav48" / name
        _txt_names = [p.stem for p in txt_folder.iterdir()]
        _wav_names = [p.stem for p in wav_folder.iterdir()]
        _names = list(set(_txt_names) & set(_wav_names))
        file_count += len(_names)

        for file_name in _names:
            spk = file_name.split("_")[0]
            if spk not in speaker_mapping.keys():
                speaker_mapping[spk] = len(speaker_mapping)
            speaker = speaker_mapping[spk]

            txt_path, wav_path = txt_folder / f"{file_name}.txt", wav_folder / f"{file_name}.wav"

            wav, sr = torchaudio.load(wav_path)
            if pauses_csv is not None:
                if not file_name in border_map.keys():
                    continue
                left_border, right_border = border_map[file_name]
                wav = wav[:, int(left_border * sr) : int(right_border * sr)]
                torchaudio.save(wav_path, wav, sr)
            duration = wav.shape[1] / sr

            with open(txt_path, "r") as txt_file:
                text = txt_file.readline().strip()

            d = {
                "audio_filepath": str(wav_path),
                "text": text,
                "duration": duration,
                "speaker": speaker,
            }
            border_file_count += 1

            data.append(d)

    print(f"Total files: {file_count}")
    print(f"Files with borders: {border_file_count}")

    train_data, dev_data = train_test_split(data, test_size=num_val_samples, random_state=SEED)

    with open(root_path / "train.json", "w") as train_file:
        for d in train_data:
            train_file.write(json.dumps(d))
            train_file.write("\n")

    with open(root_path / "dev.json", "w") as dev_file:
        for d in dev_data:
            dev_file.write(json.dumps(d))
            dev_file.write("\n")


if __name__ == "__main__":
    """
    Optional file pauses-csv
    wget https://raw.githubusercontent.com/mueller91/tts_alignments/main/vctk/vctk.csv
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--pauses-csv", type=Path)
    parser.add_argument("--num-val-samples", type=int, default=100)
    args = parser.parse_args()

    preprocess_vctk(args.root_path, args.pauses_csv, args.num_val_samples)
