import argparse
import importlib
from pathlib import Path

import torch
from omegaconf import omegaconf

DEVICE = torch.device("cuda:0")


def parse_target(target: str):
    target_parts = target.split(".")

    return ".".join(target_parts[:-1]), target_parts[-1]


def load_acoustic_model(ckpt_path: Path):
    d = torch.load(ckpt_path, map_location=DEVICE)
    conf = d["hyper_parameters"]
    with omegaconf.open_dict(conf):
        conf.pop("train_ds")
        conf.pop("validation_ds")
        conf.pop("optim")

    target_prefix, target_module = parse_target(conf.target)
    cls = getattr(importlib.import_module(target_prefix), target_module)
    model = cls(conf)
    model.load_state_dict(d["state_dict"])
    model.to(DEVICE)

    return model


def generate_spectrograms(model, output_path, input_file_path, speakers):
    with open(input_file_path, "r") as in_file:
        lines = in_file.readlines()

    output_path.mkdir(exist_ok=True)

    if speakers is not None:
        speakers = torch.tensor(speakers, dtype=torch.long, device=DEVICE)

    for i, line in enumerate(lines):
        line = line.strip()
        tokens = model.parse(line)

        print(f"For line: {i}\nWith text: {line}")
        specs = []
        with torch.no_grad():
            if speakers is None:
                spec = model.generate_spectrogram(tokens=tokens)
                specs.append(spec)
                print(f"\tSpectrogram shape: {spec.shape}")
            else:
                for speaker in speakers:
                    spec = model.generate_spectrogram(tokens=tokens, speaker_idx=speaker.unsqueeze(0))
                    specs.append(spec)
                    print(f"\tFor speaker: {speaker.item()}\n\tSpectrogram shape: {spec.shape}")

        if speaker is None:
            torch.save(spec.cpu(), output_path / f"spec_{i}.pt")
        else:
            for speaker, spec in zip(speakers, specs):
                torch.save(spec.cpu(), output_path / f"spec_{i}_speaker_{speaker.item()}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--speakers", type=int, nargs="+")
    args = parser.parse_args()

    model = load_acoustic_model(args.ckpt_path)
    print("Successfully loaded acoustic model")

    generate_spectrograms(model, args.output_path, args.input_file, args.speakers)
