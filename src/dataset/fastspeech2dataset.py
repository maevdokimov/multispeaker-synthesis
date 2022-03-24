import collections as py_collections
import json
import logging
import pickle
from os.path import expanduser
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.core.classes import Dataset
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class FastSpeech2Dataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(("B"), LengthsType()),
            "transcripts": NeuralType(("B", "T"), TokenIndex()),
            "transcript_length": NeuralType(("B"), LengthsType()),
            "durations": NeuralType(("B", "T"), TokenDurationType()),
            "pitches": NeuralType(("B", "T"), RegressionValuesType()),
            "energies": NeuralType(("B", "T"), RegressionValuesType()),
        }

    def __init__(
        self,
        manifest_filepath: str,
        mappings_filepath: str,
        sample_rate: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        ignore_file: Optional[str] = None,
        trim: bool = False,
        load_supplementary_values: bool = False,
        load_speaker: bool = False,
    ):
        """
        Dataset that loads audio, phonemes and their durations, pitches per frame, and energies per frame
        for FastSpeech 2 from paths described in a JSON manifest (see the AudioDataset documentation for details
        on the manifest format), as well as a mappings file for word to phones and phones to indices.
        The text in the manifest is ignored; instead, the phoneme indices for prediction come from the
        duration files.
        For each sample, paths for duration, energy, and pitch files are inferred from the manifest's audio
        filepaths by replacing '/wavs' with '/phoneme_durations', '/pitches', and '/energies', and swapping out
        the file extension to '.pt', '.npy', and '.npy' respectively.
        For example, given manifest audio path `/data/LJSpeech/wavs/LJ001-0001.wav`, the inferred duration and
        phonemes file path would be `/data/LJSpeech/phoneme_durations/LJ001-0001.pt`.
        Note that validation datasets only need the audio files and phoneme & duration files, set
        `load_supplementary_values` to False for validation sets.
        Args:
            manifest_filepath (str): Path to the JSON manifest file that lists audio files.
            mappings_filepath (str): Path to a JSON mappings file that contains mappings "word2phones" and
                "phone2idx". The latter is used to determine the padding index.
            sample_rate (int): Target sample rate of the audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration (float): If audio is shorter than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            ignore_file (str): Optional pickled file which contains a list of files to ignore (e.g. files that
                contain OOV words).
                Defaults to None.
            trim (bool): Whether to use librosa.effects.trim on the audio clip.
                Defaults to False.
            load_supplementary_values (bool): Whether or not to load pitch and energy files. Set this to False for
                validation datasets.
                Defaults to False.
            load_speaker (bool): Whether or not to load speaker id.
                Defaults to False.
        """
        super().__init__()

        # Retrieve mappings from file
        with open(mappings_filepath, "r") as f:
            mappings = json.load(f)
            self.word2phones = mappings["word2phones"]
            self.phone2idx = mappings["phone2idx"]

        # Load data from manifests
        audio_files = []
        total_duration = 0
        if isinstance(manifest_filepath, str):
            manifest_filepath = [manifest_filepath]
        for manifest_file in manifest_filepath:
            with open(expanduser(manifest_file), "r") as f:
                logging.info(f"Loading dataset from {manifest_file}.")
                for line in f:
                    item = json.loads(line)
                    audio_files.append(
                        {
                            "audio_filepath": item["audio_filepath"],
                            "duration": item["duration"],
                            "speaker": item["speaker"],
                        }
                    )
                    total_duration += item["duration"]

        total_dataset_len = len(audio_files)
        logging.info(f"Loaded dataset with {total_dataset_len} files totalling {total_duration/3600:.2f} hours.")
        self.data = []

        field_names = "audio_file duration text_tokens"
        if load_supplementary_values:
            field_names += " pitches energies"
        if load_speaker:
            field_names += " speaker"
        dataitem = py_collections.namedtuple(typename="AudioTextEntity", field_names=field_names)

        if ignore_file:
            logging.info(f"using {ignore_file} to prune dataset.")
            with open(ignore_file, "rb") as f:
                wavs_to_ignore = set(pickle.load(f))

        pruned_duration = 0
        pruned_items = 0
        for item in audio_files:
            audio_path = Path(item["audio_filepath"])
            utterance_id = audio_path.stem

            # Prune data according to min/max_duration & the ignore file
            if (min_duration and item["duration"] < min_duration) or (
                max_duration and item["duration"] > max_duration
            ):
                pruned_duration += item["duration"]
                pruned_items += 1
                continue
            if ignore_file and (utterance_id in wavs_to_ignore):
                pruned_items += 1
                pruned_duration += item["duration"]
                wavs_to_ignore.remove(utterance_id)
                continue

            # Root path according to structure root/audio/speaker/utterance_id.wav
            root_path = audio_path.parents[2]

            # Else not pruned, load additional info
            # Phoneme durations and text token indices from durations file
            dur_path = root_path / "phoneme_durations" / audio_path.parts[-2] / f"{audio_path.stem}.pt"
            duration_info = torch.load(dur_path)
            durs = duration_info["token_duration"]
            text_tokens = duration_info["text_encoded"]

            data_args = {
                "audio_file": item["audio_filepath"],
                "duration": durs,
                "text_tokens": text_tokens,
            }

            if load_supplementary_values:
                # Load pitch file (F0s)
                pitch_path = root_path / "pitches" / audio_path.parts[-2] / f"{audio_path.stem}.npy"
                pitches = torch.from_numpy(np.load(pitch_path).astype(dtype="float32"))

                # Load energy file (L2-norm of the amplitude of each STFT frame of an utterance)
                energies_path = root_path / "energies" / audio_path.parts[-2] / f"{audio_path.stem}.npy"
                energies = torch.from_numpy(np.load(energies_path))

                data_args["pitches"] = torch.clamp(pitches, min=1e-5)
                data_args["energies"] = energies
            if load_speaker:
                data_args["speaker"] = item["speaker"]

            self.data.append(dataitem(**data_args))

        logging.info(f"Pruned {pruned_items} files and {pruned_duration/3600:.2f} hours.")
        logging.info(
            f"Final dataset contains {len(self.data)} files and {(total_duration-pruned_duration)/3600:.2f} hours."
        )

        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate)
        self.trim = trim
        self.load_supplementary_values = load_supplementary_values
        self.load_speaker = load_speaker

    def __getitem__(self, index):
        sample = self.data[index]

        features = self.featurizer.process(sample.audio_file, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()
        t, tl = sample.text_tokens.long(), torch.tensor(len(sample.text_tokens)).long()

        pitches, energies, speaker = None, None, None
        if self.load_supplementary_values:
            pitches = sample.pitches
            energies = sample.energies
        if self.load_speaker:
            speaker = sample.speaker

        return f, fl, t, tl, sample.duration, pitches, energies, speaker

    def __len__(self):
        return len(self.data)

    def _collate_fn(self, batch):
        pad_id = len(self.phone2idx)
        if self.load_supplementary_values:
            _, audio_lengths, _, tokens_lengths, duration, pitches, energies, _ = zip(*batch)
        else:
            _, audio_lengths, _, tokens_lengths, duration, _, _, _ = zip(*batch)
        max_audio_len = 0
        max_audio_len = max(audio_lengths).item()
        max_tokens_len = max(tokens_lengths).item()
        max_durations_len = max([len(i) for i in duration])
        max_duration_sum = max([sum(i) for i in duration])
        if self.load_supplementary_values:
            max_pitches_len = max([len(i) for i in pitches])
            max_energies_len = max([len(i) for i in energies])
            if max_pitches_len != max_energies_len or max_pitches_len != max_duration_sum:
                logging.warning(
                    f"max_pitches_len: {max_pitches_len} != max_energies_len: {max_energies_len} != "
                    f"max_duration_sum:{max_duration_sum}. Your training run will error out!"
                )

        # Add padding where necessary
        audio_signal, tokens, duration_batched, pitches_batched, energies_batched = [], [], [], [], []
        for sample_tuple in batch:
            sig, sig_len, tokens_i, tokens_i_len, duration, pitch, energy, speakers = sample_tuple

            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
            tokens_i_len = tokens_i_len.item()
            if tokens_i_len < max_tokens_len:
                pad = (0, max_tokens_len - tokens_i_len)
                tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
            tokens.append(tokens_i)
            if len(duration) < max_durations_len:
                pad = (0, max_durations_len - len(duration))
                duration = torch.nn.functional.pad(duration, pad)
            duration_batched.append(duration)

            if self.load_supplementary_values:
                pitch = pitch.squeeze(0)
                if len(pitch) < max_pitches_len:
                    pad = (0, max_pitches_len - len(pitch))
                    pitch = torch.nn.functional.pad(pitch.squeeze(0), pad)
                pitches_batched.append(pitch)

                if len(energy) < max_energies_len:
                    pad = (0, max_energies_len - len(energy))
                    energy = torch.nn.functional.pad(energy, pad)
                energies_batched.append(energy)

        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
        tokens = torch.stack(tokens)
        tokens_lengths = torch.stack(tokens_lengths)
        duration_batched = torch.stack(duration_batched)

        pitches_batched, energies_batched, speakers_batched = None, None, None
        if self.load_supplementary_values:
            pitches_batched = torch.stack(pitches_batched)
            energies_batched = torch.stack(energies_batched)
            assert pitches_batched.shape == energies_batched.shape
        if self.load_speaker:
            speakers_batched = torch.tensor(speakers, dtype=torch.int32)

        return (
            audio_signal,
            audio_lengths,
            tokens,
            tokens_lengths,
            duration_batched,
            pitches_batched,
            energies_batched,
            speakers_batched,
        )
