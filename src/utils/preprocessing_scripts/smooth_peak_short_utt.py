###
# At hi_fi_tts dataset we have sharp peak at utterances
# with length 1-3 (like 2/5 of the data)
# The purpose of this script is to drop some short utteances
# to support balance of utterance lengths
###

import argparse
import json
import random
from pathlib import Path

import numpy as np


def smooth_data(file_path: Path, out_path: Path, smoothing_peak: float, n_bins: int):
    dicts, filtered_dicts = [], []

    with open(file_path, "r") as file:
        for line in file:
            d = json.loads(line)
            dicts.append(d)

    print(f"Dataset size: {len(dicts)}")

    # First let's see how much samples we have
    # at our new distribution peak
    peak_samples = len(
        [
            d
            for d in dicts
            if d["duration"] >= smoothing_peak and d["duration"] < smoothing_peak + smoothing_peak / n_bins
        ]
    )
    print(f"Peak samples: {peak_samples}")

    # We grab 8 bins and linearly increase their size
    bin_borders = np.linspace(0, smoothing_peak, n_bins + 1)
    bin_sizes = np.linspace(len(dicts) // 400, peak_samples, n_bins + 1).astype(np.int32)
    print(f"Bin borders: {bin_borders}")
    print(f"Bin sizes: {bin_sizes}")

    for i in range(n_bins):
        print("----------------------")
        print(f"Bin borders step {i}")
        print(bin_borders[i], bin_borders[i + 1])
        _dicts = [d for d in dicts if d["duration"] >= bin_borders[i] and d["duration"] < bin_borders[i + 1]]
        print(f"Num samples: {len(_dicts)}")
        filtered_dicts.extend(random.sample(_dicts, min(len(_dicts), bin_sizes[i])))

    filtered_dicts.extend([d for d in dicts if d["duration"] >= smoothing_peak])

    with open(out_path, "w") as file:
        for d in filtered_dicts:
            file.write(json.dumps(d))
            file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-path", type=Path, required=True)
    parser.add_argument("--out-path", type=Path, required=True)
    parser.add_argument("--smoothing-peak", type=float, default=4)
    parser.add_argument("--n-bins", type=float, default=8)
    args = parser.parse_args()

    smooth_data(args.file_path, args.out_path, args.smoothing_peak, args.n_bins)
