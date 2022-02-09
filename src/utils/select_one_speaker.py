import argparse
import json
import pickle
from pathlib import Path


def select_one_speaker(file_path: Path, out_path: Path, mapping_path: Path, speaker_id: str):
    with open(mapping_path, "rb") as mapping_file:
        mapping = pickle.load(mapping_file)
    speaker = mapping[speaker_id]

    with open(file_path, "r") as in_file, open(out_path, "w") as out_file:
        for line in in_file:
            d = json.loads(line)
            if d["speaker"] == speaker:
                del d["speaker"]
                out_file.write(f"{json.dumps(d)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--speaker-id", required=True)
    args = parser.parse_args()

    select_one_speaker(args.file_path, args.out_path, args.mapping_path, args.speaker_id)
