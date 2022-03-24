import argparse
import json
import multiprocessing as mp
from pathlib import Path

import librosa
import numpy as np
from omegaconf import OmegaConf


def _process_audio(path, root_path, config):
    speaker_dir = path.parts[-2]
    pitches_spk_dir = root_path / "pitches" / speaker_dir
    energies_spk_dir = root_path / "energies" / speaker_dir
    if not (pitches_spk_dir).exists():
        pitches_spk_dir.mkdir()
    if not (energies_spk_dir).exists():
        energies_spk_dir.mkdir()

    pitch_path = root_path / "pitches" / speaker_dir / f"{path.stem}.npy"
    energy_path = root_path / "energies" / speaker_dir / f"{path.stem}.npy"

    audio, _ = librosa.load(path, sr=config.sampling_rate)

    f0, _, _ = librosa.pyin(
        audio,
        fmin=config.fmin,
        fmax=config.fmax,
        frame_length=config.frame_length,
        sr=config.sampling_rate,
        fill_na=0.0,
    )
    stft_amplitude = np.abs(librosa.stft(audio, n_fft=config.n_fft, hop_length=config.hop_length))
    energy = np.linalg.norm(stft_amplitude, axis=0)

    assert f0.shape == energy.shape
    np.save(pitch_path, f0)
    np.save(energy_path, energy)


def extract_pitch_energy(manifest_path: Path, root_path: Path, config_path: Path, num_workers: int):
    cfg = OmegaConf.load(config_path)

    data_paths = []
    with open(manifest_path, "r") as file:
        for line in file:
            line = line.strip()
            d = json.loads(line)
            data_paths.append(Path(d["audio_filepath"]))

    (root_path / "pitches").mkdir(exist_ok=True)
    (root_path / "energies").mkdir(exist_ok=True)

    args = [(path, root_path, cfg) for path in data_paths]

    pool = mp.Pool(num_workers)
    for i, idx in enumerate(range(0, len(data_paths), num_workers)):
        if i % 100 == 0:
            print(f"Processed {idx} samples of {len(data_paths)}")
        pool.starmap(_process_audio, args[idx : idx + num_workers])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=6)
    args = parser.parse_args()

    extract_pitch_energy(args.manifest_path, args.root_path, args.config_path, args.num_workers)
