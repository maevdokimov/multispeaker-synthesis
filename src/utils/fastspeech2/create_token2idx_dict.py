import argparse
import json
from pathlib import Path


def create_token2idx_dict(dict_path: Path, mapping_path: Path):
    phonemes = set()
    word2phones = {}
    with open(dict_path, "r") as f:
        for line in f:
            line = line.split()
            word = line[0]
            tokens = line[1:]

            word2phones[word] = tokens
            phonemes.update(tokens)

    word2phones[","] = [" "]
    word2phones[";"] = [" "]
    word2phones["."] = [" "]
    word2phones["!"] = [" "]
    word2phones["?"] = [" "]
    word2phones['"'] = [" "]
    word2phones["-"] = [" "]

    phonemes = sorted(list(phonemes))
    phone2idx = {k: i for i, k in enumerate(phonemes)}
    phone2idx[" "] = len(phone2idx)
    phone2idx["sil"] = phone2idx[" "]
    phone2idx["sp"] = phone2idx[" "]
    phone2idx["spn"] = phone2idx[" "]

    dicts = {
        "phone2idx": phone2idx,
        "word2phones": word2phones,
    }
    with open(mapping_path, "w") as f:
        json.dump(dicts, f)

    print(f"Total number of phone indices: {len(phone2idx)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict-path", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    args = parser.parse_args()

    create_token2idx_dict(args.dict_path, args.mapping_path)
