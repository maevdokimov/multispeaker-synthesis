import argparse
import json
import pickle
from pathlib import Path


def prepare_speaker_counts(manifest_path: Path, output_path: Path):
    speakers = {}

    with open(manifest_path, "r") as in_file:
        for line in in_file:
            line = line.strip()
            d = json.loads(line)
            speakers[d["speaker"]] = speakers.get(d["speaker"], 0) + 1

    num_speakers = max(speakers.keys()) + 1
    print(f"Num speakers: {num_speakers}")

    for i in range(num_speakers):
        if i not in speakers.keys():
            print(f"Speaker {i} is missing samples. Setting counter to 0")
            speakers[i] = 0

    with open(output_path, "wb") as out_file:
        pickle.dump(speakers, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    prepare_speaker_counts(args.manifest_path, args.output_path)
