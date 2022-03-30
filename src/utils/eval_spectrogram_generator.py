import argparse
import importlib
from pathlib import Path

import torch
import torchaudio
from omegaconf import omegaconf

DEVICE = torch.device("cuda:0")
REMOVE_KEYS = ["train_ds", "validation_ds", "optim", "sched"]


def parse_target(target: str):
    target_parts = target.split(".")

    return ".".join(target_parts[:-1]), target_parts[-1]


def print_config_diff(conf_acoustic, conf_vocoder):
    acoustic_keys, vocoder_keys = list(conf_acoustic.keys()), list(conf_vocoder.keys())
    intersection_keys = [key for key in acoustic_keys if key in vocoder_keys]

    print("---------------- Intersected keys ----------------")
    for key in intersection_keys:
        print(f"Key: {key}, acoustic model: {conf_acoustic[key]}, vocoder model: {conf_vocoder[key]}")

    print("---------------- Acoustic model keys ----------------")
    for key in [key for key in acoustic_keys if key not in intersection_keys]:
        print(f"Key: {key}, value: {acoustic_model[key]}")

    print("---------------- Vocoder keys ----------------")
    for key in [key for key in vocoder_keys if key not in intersection_keys]:
        print(f"Key: {key}, value: {conf_vocoder[key]}")


def load_nemo_checkpoint(ckpt_path: Path):
    d = torch.load(ckpt_path, map_location=DEVICE)
    conf = d["hyper_parameters"]

    with omegaconf.open_dict(conf):
        for key in REMOVE_KEYS:
            if key in conf.keys():
                conf.pop(key)

    target_prefix, target_module = parse_target(conf.target)
    cls = getattr(importlib.import_module(target_prefix), target_module)
    model = cls(conf)
    model.load_state_dict(d["state_dict"])
    model.to(DEVICE)

    return model, conf.preprocessor


def generate_spectrograms(acoustic_model, output_path, input_file_path, speakers):
    with open(input_file_path, "r") as in_file:
        lines = in_file.readlines()

    output_path.mkdir(exist_ok=True)

    if speakers is not None:
        speakers = torch.tensor(speakers, dtype=torch.long, device=DEVICE)

    for i, line in enumerate(lines):
        line = line.strip()
        tokens = acoustic_model.parse(line)

        print(f"For line: {i}\nWith text: {line}")
        specs = []
        with torch.no_grad():
            if speakers is None:
                spec = acoustic_model.generate_spectrogram(tokens=tokens)
                specs.append(spec)
                print(f"\tSpectrogram shape: {spec.shape}")
            else:
                for speaker in speakers:
                    spec = acoustic_model.generate_spectrogram(tokens=tokens, speaker_idx=speaker.unsqueeze(0))
                    specs.append(spec)
                    print(f"\tFor speaker: {speaker.item()}\n\tSpectrogram shape: {spec.shape}")

        if speaker is None:
            torch.save(spec.cpu(), output_path / f"spec_{i}.pt")
        else:
            for speaker, spec in zip(speakers, specs):
                torch.save(spec.cpu(), output_path / f"spec_{i}_speaker_{speaker.item()}.pt")


def eval_pipeline(acoustic_model, vocoder, output_path, input_file_path, sample_rate, speakers):
    with open(input_file_path, "r") as in_file:
        lines = in_file.readlines()

    output_path.mkdir(exist_ok=True)

    if speakers is not None:
        speakers = torch.tensor(speakers, dtype=torch.long, device=DEVICE)

    for i, line in enumerate(lines):
        line = line.strip()
        tokens = acoustic_model.parse(line)

        print(f"For line: {i}\nWith text: {line}")
        specs, wavs = [], []
        with torch.no_grad():
            if speakers is None:
                spec = acoustic_model.generate_spectrogram(tokens=tokens)
                wav = vocoder(spec=spec)
                specs.append(spec)
                wavs.append(wav)
                print(f"\tSpectrogram shape: {spec.shape}, Wav shape: {wav.shape}")
            else:
                for speaker in speakers:
                    spec = acoustic_model.generate_spectrogram(tokens=tokens, speaker_idx=speaker.unsqueeze(0))
                    wav = vocoder(spec=spec)
                    specs.append(spec)
                    wavs.append(wav)
                    print(
                        f"\tFor speaker: {speaker.item()}\n\tSpectrogram shape: {spec.shape}, Wav shape: {wav.shape}"
                    )

        if speakers is None:
            torch.save(specs[0].cpu(), output_path / f"spec_{i}.pt")
            torchaudio.save(output_path / f"audio_{i}.wav", wavs[0].cpu().squeeze(0), sample_rate)
        else:
            for speaker, spec, wav in zip(speakers, specs, wavs):
                torch.save(spec.cpu(), output_path / f"spec_{i}_speaker_{speaker.item()}.pt")
                torchaudio.save(
                    output_path / f"audio_{i}_speaker_{speaker.item()}.wav", wav.cpu().squeeze(0), sample_rate
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acoustic-ckpt-path", type=Path, required=True)
    parser.add_argument("--vocoder-ckpt-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--speakers", type=int, nargs="+")
    args = parser.parse_args()

    acoustic_model, conf_acoustic = load_nemo_checkpoint(args.acoustic_ckpt_path)
    print("Successfully loaded acoustic model")
    vocoder, conf_vocoder = load_nemo_checkpoint(args.vocoder_ckpt_path)
    print("Successfully loaded vocoder")

    print_config_diff(conf_acoustic, conf_vocoder)

    eval_pipeline(acoustic_model, vocoder, args.output_path, args.input_file, conf_vocoder.sample_rate, args.speakers)
