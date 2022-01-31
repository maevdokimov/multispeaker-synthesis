import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, Optional

from tqdm import tqdm


def process_hifitts(root_path: Path, mapping: Optional[Dict]):
    train_list, dev_list = [], []
    new_map = mapping is None
    mapping = {} if mapping is None else mapping

    for p in root_path.iterdir():
        if "train" in p.stem and p.suffix == ".json":
            train_list.append(p)
        elif "dev" in p.stem and p.suffix == ".json":
            dev_list.append(p)

    with open(root_path / "train.json", "w") as out_file:
        for p in tqdm(train_list):
            spk = p.stem.split("_")[0]
            if not spk in mapping.keys() and new_map:
                mapping[spk] = len(mapping)
            elif not spk in mapping.keys():
                raise ValueError(f"No such speaker {spk} in {mapping}")

            with open(p, "r") as in_file:
                for line in in_file:
                    line = line.strip()
                    d = json.loads(line)
                    d["speaker"] = mapping[spk]
                    d["audio_filepath"] = str((root_path / d["audio_filepath"]).resolve())
                    out_file.write(f"{json.dumps(d)}\n")

    with open(root_path / "dev.json", "w") as out_file:
        for p in tqdm(dev_list):
            spk = p.stem.split("_")[0]
            if not spk in mapping.keys():
                continue

            with open(p, "r") as in_file:
                for line in in_file:
                    line = line.strip()
                    d = json.loads(line)
                    d["speaker"] = mapping[spk]
                    d["audio_filepath"] = str((root_path / d["audio_filepath"]).resolve())
                    out_file.write(f"{json.dumps(d)}\n")

    if new_map:
        with open(root_path / "mapping.pkl", "wb") as out_file:
            pickle.dump(mapping, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--speaker-map", type=Path)
    args = parser.parse_args()

    spk_map = None
    if "speaker_map" in args:
        with open(args.speaker_map, "rb") as spk_map_file:
            spk_map = pickle.load(spk_map_file)
    process_hifitts(args.root_path, spk_map)
