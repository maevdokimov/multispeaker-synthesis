import argparse
import json
from pathlib import Path


def concat_datasets(first_path: Path, second_path: Path, out_path: Path):
    first_lines, second_lines = [], []

    with open(first_path, "r") as first_file:
        for line in first_file:
            d = json.loads(line)
            first_lines.append(d)

    with open(second_path, "r") as second_file:
        for line in second_file:
            d = json.loads(line)
            second_lines.append(d)

    first_has_speaker = "speaker" in first_lines[0].keys()
    second_has_speaker = "speaker" in second_lines[0].keys()

    first_num_speakers, second_num_speakers = None, None
    if first_has_speaker:
        first_num_speakers = len(set([d["speaker"] for d in first_lines]))
        print(f"First num speakers: {first_num_speakers}")

    if second_has_speaker:
        second_num_speakers = len(set([d["speaker"] for d in second_lines]))
        print(f"Second num speakers: {second_num_speakers}")

    if not first_has_speaker and not second_has_speaker:
        for d in first_lines:
            d["speaker"] = 0
        for d in second_lines:
            d["speaker"] = 1
    elif first_has_speaker and not second_has_speaker:
        for d in second_lines:
            d["speaker"] = first_num_speakers
    elif first_has_speaker and second_has_speaker:
        for d in second_lines:
            d["speaker"] += first_num_speakers

    with open(out_path, "w") as out_file:
        for d in first_lines:
            out_file.write(f"{json.dumps(d)}\n")
        for d in second_lines:
            out_file.write(f"{json.dumps(d)}\n")

    return first_num_speakers, second_num_speakers


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-path", type=Path, required=True)
    parser.add_argument("--second-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    args = parser.parse_args()

    concat_datasets(args.first_path, args.second_path, args.out_path)
