import argparse
import json
from pathlib import Path

from nemo.collections.common.parts.preprocessing import parsers
from tqdm import tqdm


def create_txt_transcripts(manifest_path: Path, text_type: str):
    lines = {}
    with open(manifest_path, "r") as manifest_file:
        for line in manifest_file:
            line = line.strip()
            d = json.loads(line)

            lines[d["audio_filepath"]] = d[text_type]

    for path, text in tqdm(lines.items()):
        path = Path(path).with_suffix(".txt")
        norm_text = parsers.make_parser(name="en")._normalize(text)
        with open(path, "w") as f_txt:
            f_txt.write(norm_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--text-type", type=str, default="text")
    args = parser.parse_args()

    create_txt_transcripts(args.manifest_path, args.text_type)
