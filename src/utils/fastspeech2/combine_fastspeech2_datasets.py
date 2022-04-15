import argparse
import pickle
import shutil
from pathlib import Path

from src.utils.preprocessing_scripts.concat_datasets import concat_datasets


def combine_datasets(first_root: Path, second_root: Path, output_path: Path):
    concat_datasets(first_root / "train.json", second_root / "train.json", output_path / "train.json")
    concat_datasets(first_root / "dev.json", second_root / "dev.json", output_path / "dev.json")

    first_oov, second_oov = first_root / "wavs_to_ignore.pkl", second_root / "wavs_to_ignore.pkl"
    with open(first_oov, "rb") as first_file, open(second_oov, "rb") as second_file:
        _first_file, _second_file = pickle.load(first_file), pickle.load(second_file)
        combined_wavs_to_ignore = _first_file + _second_file
        with open(output_path / "wavs_to_ignore.pkl", "wb") as out_file:
            pickle.dump(combined_wavs_to_ignore, out_file)

    shutil.copy(first_root / "mappings.json", output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-root", type=Path, required=True)
    parser.add_argument("--second-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    combine_datasets(args.first_root, args.second_root, args.output_path)
