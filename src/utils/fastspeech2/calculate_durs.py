import argparse
import json
from math import ceil
from pathlib import Path

import numpy as np
import tgt
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def get_textgrids(root_path: Path):
    """Temporary solution for LJspeech"""
    textgrids = []
    alignments_path = root_path / "alignments"
    for spk_path in alignments_path.iterdir():
        for tg_path in spk_path.iterdir():
            textgrids.append(tg_path)

    return textgrids


def create_target_path(target_dir: Path, base_path: Path):
    """Temporary solution for LJspeech"""
    pass


def _calculate_durations(textgrid, phone2idx, config):
    tokens = []
    durs = []

    frames_per_second = config.sampling_rate / config.hop_length
    tg = tgt.read_textgrid(textgrid, include_empty_intervals=True)
    data_tier = tg.get_tier_by_name("phones")

    total_frames = ceil((data_tier.end_time - data_tier.start_time) * frames_per_second)

    se_in_frames = np.array([(frames_per_second * d.start_time, frames_per_second * d.end_time) for d in data_tier])
    _se_in_frames = np.round(se_in_frames)
    durs = (_se_in_frames[:, 1] - _se_in_frames[:, 0]).astype(int)
    blank_set = ("sil", "sp", "spn", "", "<unk>")
    blank_token = " "

    tokens, durations = [], []
    for i in range(len(data_tier)):
        x = data_tier[i].text
        if x == "spn":
            return None, None, None
        x = blank_token if x in blank_set else x

        if len(tokens) and tokens[-1] == blank_token and x == blank_token:
            durations[-1] += durs[i]
        else:
            tokens.append(x)
            durations.append(durs[i])

    tokens_enc = [phone2idx[token] for token in tokens]
    tokens_enc, durations = torch.LongTensor(tokens_enc), torch.LongTensor(durations)

    # Add rounding error to final token
    durations[-1] += total_frames - durations.sum()

    return tokens, tokens_enc, durations


def calculate_durations(root_path: Path, mapping_path: Path, config_path: Path):
    textgrid_list = get_textgrids(root_path)

    target_dir = root_path / "phoneme_durations"
    target_dir.mkdir(exist_ok=True)

    phone2idx = None
    with open(mapping_path, "r") as f:
        mappings = json.load(f)
        phone2idx = mappings["phone2idx"]

    oov_samples = []

    config = OmegaConf.load(config_path)
    for textgrid in tqdm(textgrid_list):
        phones_mfa, tokens_mfa, durs = _calculate_durations(textgrid, phone2idx, config)

        if phones_mfa is None:
            oov_samples.append(textgrid)
            continue

        target_path = create_target_path(target_dir, textgrid)
        torch.save({"text_encoded": tokens_mfa, "token_duration": durs}, target_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-path", type=Path, required=True)
    parser.add_argument("--mapping-path", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    args = parser.parse_args()

    calculate_durations(args.root_path, args.mapping_path, args.config_path)
