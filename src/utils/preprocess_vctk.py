import argparse
import json
from pathlib import Path

import torchaudio
from sklearn.model_selection import train_test_split

SEED = 0xDEADF00D


def preprocess_vctk(root_path: Path, num_val_samples: int = 100):
    txt_folder_names = [p.stem for p in (root_path / "txt").iterdir()]
    wav_folder_names = [p.stem for p in (root_path / "wav48").iterdir()]
    folder_names = list(set(txt_folder_names) & set(wav_folder_names))

    speaker_mapping = {}
    data = []

    for name in folder_names:
        txt_folder, wav_folder = root_path / "txt" / name, root_path / "wav48" / name
        _txt_names = [p.stem for p in txt_folder.iterdir()]
        _wav_names = [p.stem for p in wav_folder.iterdir()]
        _names = list(set(_txt_names) & set(_wav_names))

        for file_name in _names:
            spk = file_name.split("_")[0]
            if spk not in speaker_mapping.keys():
                speaker_mapping[spk] = len(speaker_mapping)
            speaker = speaker_mapping[spk]

            txt_path, wav_path = txt_folder / f"{file_name}.txt", wav_folder / f"{file_name}.wav"

            wav, sr = torchaudio.load(wav_path)
            duration = wav.shape[1] / sr

            with open(txt_path, "r") as txt_file:
                text = txt_file.readline().strip()

            d = {
                "audio_filepath": str(wav_path),
                "text": text,
                "duration": duration,
                "speaker": speaker,
            }

            data.append(d)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--num-val-samples", type=int, default=100)
    args = parser.parse_args()

    preprocess_vctk(args.root_path, args.num_val_samples)
