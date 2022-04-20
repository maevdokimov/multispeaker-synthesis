import argparse
import importlib
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from omegaconf import OmegaConf, omegaconf

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
        print(f"Key: {key}, value: {conf_acoustic[key]}")

    print("---------------- Vocoder keys ----------------")
    for key in [key for key in vocoder_keys if key not in intersection_keys]:
        print(f"Key: {key}, value: {conf_vocoder[key]}")


def load_nemo_checkpoint(ckpt_path: Path, config_override_path: Optional[Path]):
    d = torch.load(ckpt_path, map_location=DEVICE)
    conf = d["hyper_parameters"]

    if config_override_path is not None:
        config_override = OmegaConf.load(config_override_path)
        conf = OmegaConf.merge(conf, config_override)

    with omegaconf.open_dict(conf):
        for key in REMOVE_KEYS:
            if key in conf.keys():
                conf.pop(key)

    target_prefix, target_module = parse_target(conf.target)
    cls = getattr(importlib.import_module(target_prefix), target_module)
    model = cls(conf)
    model.load_state_dict(d["state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, conf.preprocessor


def generate_spectrograms(acoustic_model, output_path, input_file_path, speakers):
    with open(input_file_path, "r") as in_file:
        lines = in_file.readlines()

    output_path.mkdir(exist_ok=True)

    if speakers is not None:
        speakers = torch.tensor(speakers, dtype=torch.long, device=DEVICE)

    miss_cnt = 0
    for i, line in enumerate(lines):
        line = line.strip()
        print(f"For line: {i}\nWith text: {line}")

        try:
            tokens = acoustic_model.parse(line)
        except Exception:
            miss_cnt += 1
            continue
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

        if speakers is None:
            torch.save(spec.cpu(), output_path / f"spec_{i}.pt")
        else:
            for speaker, spec in zip(speakers, specs):
                torch.save(spec.cpu(), output_path / f"spec_{i}_speaker_{speaker.item()}.pt")

    print(f"Total sentences missed: {miss_cnt}")


def eval_pipeline(acoustic_model, vocoder, output_path, input_file_path, sample_rate, speakers):
    with open(input_file_path, "r") as in_file:
        lines = in_file.readlines()

    output_path.mkdir(exist_ok=True)

    if speakers is not None:
        speakers = torch.tensor(speakers, dtype=torch.long, device=DEVICE)

    miss_cnt = 0
    for i, line in enumerate(lines):
        line = line.strip()
        print(f"For line: {i}\nWith text: {line}")

        try:
            tokens = acoustic_model.parse(line)
        except Exception:
            miss_cnt += 1
            continue
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

    print(f"Total sentences missed: {miss_cnt}")


def run_evaluation(args: argparse.Namespace):
    if args.evaluation_mode == "acoustic":
        print("Evaluating acoustic model")

        acoustic_model, conf_acoustic = load_nemo_checkpoint(args.acoustic_ckpt_path, args.override_acoustic_config)
        print("Successfully loaded acoustic model")

        print("---------------- Acoustic model keys ----------------")
        print(OmegaConf.to_yaml(conf_acoustic))

        generate_spectrograms(acoustic_model, args.output_path, args.input_file, args.speakers)
    elif args.evaluation_mode == "pipeline":
        print("Evaluating pipeline")

        acoustic_model, conf_acoustic = load_nemo_checkpoint(args.acoustic_ckpt_path, args.override_acoustic_config)
        print("Successfully loaded acoustic model")
        vocoder, conf_vocoder = load_nemo_checkpoint(args.vocoder_ckpt_path, args.override_vocoder_config)
        print("Successfully loaded vocoder")

        print_config_diff(conf_acoustic, conf_vocoder)

        eval_pipeline(
            acoustic_model, vocoder, args.output_path, args.input_file, conf_vocoder.sample_rate, args.speakers
        )
    else:
        print(f"Unknown option {args.evaluation_mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-mode", required=True, choices=["acoustic", "pipeline"])
    parser.add_argument("--acoustic-ckpt-path", type=Path, required=True)
    parser.add_argument("--vocoder-ckpt-path", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--speakers", type=int, nargs="+")
    parser.add_argument("--override-acoustic-config", type=Path)
    parser.add_argument("--override-vocoder-config", type=Path)
    args = parser.parse_args()

    run_evaluation(args)
