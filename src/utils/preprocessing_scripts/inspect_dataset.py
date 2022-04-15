import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def inspect_dataset(file_path: Path):
    chars, chars_normalized, chars_no_preprocessing = Counter(), Counter(), Counter()
    file_cnt = 0
    text_cnt = 0
    text_normalized_cnt = 0
    text_no_preprocessing_cnt = 0
    durations, speakers = [], []

    with open(file_path, "r") as file:
        for line in file:
            d = json.loads(line)
            file_cnt += 1
            durations.append(d["duration"])
            speakers.append(d["speaker"])

            if "text" in d.keys():
                chars.update(d["text"])
                text_cnt += 1
            if "text_normalized" in d.keys():
                chars_normalized.update(d["text_normalized"])
                text_normalized_cnt += 1
            if "text_no_preprocessing" in d.keys():
                chars_no_preprocessing.update(d["text_no_preprocessing"])
                text_no_preprocessing_cnt += 1

    print(f"Num lines: {file_cnt}")
    print("----------Text characters----------")
    print(f"Num texts: {text_cnt}")
    print(chars)
    print("----------Text characters----------")
    print(f"Num normalized_texts: {text_normalized_cnt}")
    print(chars_normalized)
    print("----------Text characters----------")
    print(f"Num texts no preprocessing: {text_no_preprocessing_cnt}")
    print(chars_no_preprocessing)

    print("----------Normalized characters----------")
    char_list = sorted(list(chars_normalized.keys()))
    print(f"Num normalized characters: {len(char_list)}")
    print(char_list)

    print("----------Durations----------")
    print(f"Mean: {np.mean(durations)}, std: {np.std(durations)}")
    print(f"Min: {np.min(durations)}, max: {np.max(durations)}")
    print(f"99 percentile: {np.quantile(durations, 0.99)}")
    print(f"95 percentile: {np.quantile(durations, 0.95)}")
    print(f"90 percentile: {np.quantile(durations, 0.90)}")
    print(f"10 percentile: {np.quantile(durations, 0.10)}")
    print(f"5 percentile: {np.quantile(durations, 0.05)}")
    print(f"1 percentile: {np.quantile(durations, 0.01)}")

    print("----------Speakers----------")
    unique_speakers = np.unique(speakers)
    print(f"Speakers: {unique_speakers}")
    print(f"Num speakers: {len(unique_speakers)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=Path, required=True)
    args = parser.parse_args()

    inspect_dataset(args.file_path)
