import random
from typing import Callable, Dict, List, Optional, Union

import torch
from nemo.collections.asr.parts.preprocessing.features import WaveformFeaturizer
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset
from nemo.core.neural_types import *
from nemo.utils import logging

__all__ = [
    "AudioToCharDataset",
    "AudioToCharWithDursF0Dataset",
    "AudioToCharWithPriorDataset",
    "AudioToBPEDataset",
    "TarredAudioToCharDataset",
    "TarredAudioToBPEDataset",
]


def _speech_collate_fn(batch, pad_id, has_speaker_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    packed_batch = list(zip(*batch))
    if len(packed_batch) == 6:
        _, audio_lengths, _, tokens_lengths, _, sample_ids = packed_batch
    if len(packed_batch) == 5 and has_speaker_id:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths, _ = packed_batch
    elif len(packed_batch) == 5:
        _, audio_lengths, _, tokens_lengths, sample_ids = packed_batch
    elif len(packed_batch) == 4:
        sample_ids = None
        _, audio_lengths, _, tokens_lengths = packed_batch
    else:
        raise ValueError("Expects 6, 5 or 4 tensors in the batch!")
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    if has_speaker_id:
        speakers = []
    for b in batch:
        if len(b) == 6:
            sig, sig_len, tokens_i, tokens_i_len, speaker_id, _ = b
        if len(packed_batch) == 5 and has_speaker_id:
            sig, sig_len, tokens_i, tokens_i_len, speaker_id = b
        elif len(packed_batch) == 5:
            sig, sig_len, tokens_i, tokens_i_len, _ = b
        elif len(packed_batch) == 4:
            sig, sig_len, tokens_i, tokens_i_len = b

        if has_audio:
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
        if has_speaker_id:
            speakers.append(speaker_id)

    if has_speaker_id:
        speakers = torch.stack(speakers)
    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)
    if sample_ids is None and not has_speaker_id:
        return audio_signal, audio_lengths, tokens, tokens_lengths
    elif sample_ids is None:
        speakers = torch.tensor(speakers, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, speakers
    elif not has_speaker_id:
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, sample_ids
    else:
        speakers = torch.tensor(speakers, dtype=torch.int32)
        sample_ids = torch.tensor(sample_ids, dtype=torch.int32)
        return audio_signal, audio_lengths, tokens, tokens_lengths, speakers, sample_ids


class ASRManifestProcessor:
    """
    Class that processes a manifest json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        parser: Str for a language specific preprocessor or a callable.
        max_duration: If audio exceeds this length, do not include in dataset.
        min_duration: If audio is less than this length, do not include in dataset.
        max_utts: Limit number of utterances.
        bos_id: Id of beginning of sequence symbol to append if not None.
        eos_id: Id of end of sequence symbol to append if not None.
        pad_id: Id of pad symbol. Defaults to 0.
    """

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
    ):
        self.parser = parser

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath,
            parser=parser,
            min_duration=min_duration,
            max_duration=max_duration,
            max_number=max_utts,
        )

        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

    def process_text(self, index) -> (List[int], int):
        sample = self.collection[index]

        t, tl = sample.text_tokens, len(sample.text_tokens)

        if self.bos_id is not None:
            t = [self.bos_id] + t
            tl += 1
        if self.eos_id is not None:
            t = t + [self.eos_id]
            tl += 1

        return t, tl


class _AudioTextDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations (in seconds).
    Each new line is a different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath": "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor object used to augment loaded
            audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
        return_speaker_id (bool): whether to return the speaker_id as a part of each sample
        target_speakers: list of speakers, for which we want to use full dataset
        finetune_data_percentage: percent of samples of non-target speakers to mix with data of
            target speakers
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
            "transcripts": NeuralType(("B", "T"), LabelsType()),
            "transcript_length": NeuralType(tuple("B"), LengthsType()),
            "speaker_id": NeuralType(tuple("B"), LabelsType(), optional=True),
            "sample_id": NeuralType(tuple("B"), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        parser: Union[str, Callable],
        sample_rate: int,
        int_values: bool = False,
        augmentor: "nemo.collections.asr.parts.perturb.AudioAugmentor" = None,
        max_duration: Optional[int] = None,
        min_duration: Optional[int] = None,
        max_utts: int = 0,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        return_sample_id: bool = False,
        return_speaker_id: bool = False,
        target_speakers: List[int] = None,
        finetune_data_percentage: float = None,
    ):
        self.manifest_processor = ASRManifestProcessor(
            manifest_filepath=manifest_filepath,
            parser=parser,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
        )
        self.featurizer = WaveformFeaturizer(sample_rate=sample_rate, int_values=int_values, augmentor=augmentor)
        self.trim = trim
        self.return_sample_id = return_sample_id
        self.return_speaker_id = return_speaker_id

        self.target_speakers = target_speakers
        self.finetune_data_percentage = finetune_data_percentage
        self.samples_finetune = self.finetune_data_percentage is not None
        if self.samples_finetune:
            self.target_idx, self.non_target_idx = [], []
            for i, sample in enumerate(self.manifest_processor.collection):
                if sample.speaker in target_speakers:
                    self.target_idx.append(i)
                else:
                    self.non_target_idx.append(i)

            self.dataset_size = int(len(self.target_idx) + finetune_data_percentage * len(self.non_target_idx))
            logging.info(
                f"Number of target samples: {len(self.target_idx)}; "
                f"Number of non-target samples: {len(self.non_target_idx)}; "
                f"Resultig dataset size: {self.dataset_size}"
            )

    def get_manifest_sample(self, sample_id):
        return self.manifest_processor.collection[sample_id]

    def __getitem__(self, index):
        if not self.samples_finetune:
            return self.base_getitem(index)

        if index < len(self.target_idx):
            return self.base_getitem(self.target_idx[index])
        else:
            return self.base_getitem(random.choice(self.non_target_idx))

    def base_getitem(self, index):
        sample = self.manifest_processor.collection[index]
        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(
            sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim, orig_sr=sample.orig_sr
        )
        f, fl = features, torch.tensor(features.shape[0]).long()

        t, tl = self.manifest_processor.process_text(index)

        if self.return_sample_id and self.return_speaker_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), torch.tensor(sample.speaker).long(), index
        elif self.return_speaker_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), torch.tensor(sample.speaker).long()
        elif self.return_sample_id:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long(), index
        else:
            output = f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

        return output

    def __len__(self):
        if not self.samples_finetune:
            return len(self.manifest_processor.collection)
        else:
            return self.dataset_size

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=self.manifest_processor.pad_id, has_speaker_id=self.return_speaker_id)


class AudioToCharDataset(_AudioTextDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, transcripts, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio.wav", "text_filepath":
    "/path/to/audio.txt", "duration": 23.147}
    ...
    {"audio_filepath": "/path/to/audio.wav", "text": "the
    transcription", "offset": 301.75, "duration": 0.82, "utt":
    "utterance_id", "ctm_utt": "en_4156", "side": "A"}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels: String containing all the possible characters to map to
        sample_rate (int): Sample rate to resample loaded audio to
        int_values (bool): If true, load samples as 32-bit integers. Defauts to False.
        augmentor (nemo.collections.asr.parts.perturb.AudioAugmentor): An AudioAugmentor
            object used to augment loaded audio
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        max_utts: Limit number of utterances
        blank_index: blank character index, default = -1
        unk_index: unk_character index, default = -1
        normalize: whether to normalize transcript text (default): True
        bos_id: Id of beginning of sequence symbol to append if not None
        eos_id: Id of end of sequence symbol to append if not None
        return_sample_id (bool): whether to return the sample_id as a part of each sample
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
            "transcripts": NeuralType(("B", "T"), LabelsType()),
            "transcript_length": NeuralType(tuple("B"), LengthsType()),
            "sample_id": NeuralType(tuple("B"), LengthsType(), optional=True),
        }

    def __init__(
        self,
        manifest_filepath: str,
        labels: Union[str, List[str]],
        sample_rate: int,
        int_values: bool = False,
        augmentor: "nemo.collections.asr.parts.perturb.AudioAugmentor" = None,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        max_utts: int = 0,
        blank_index: int = -1,
        unk_index: int = -1,
        normalize: bool = True,
        trim: bool = False,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
        pad_id: int = 0,
        parser: Union[str, Callable] = "en",
        return_sample_id: bool = False,
        return_speaker_id: bool = False,
        target_speakers: List[int] = None,
        finetune_data_percentage: float = None,
    ):
        self.labels = labels

        parser = parsers.make_parser(
            labels=labels, name=parser, unk_id=unk_index, blank_id=blank_index, do_normalize=normalize
        )

        super().__init__(
            manifest_filepath=manifest_filepath,
            parser=parser,
            sample_rate=sample_rate,
            int_values=int_values,
            augmentor=augmentor,
            max_duration=max_duration,
            min_duration=min_duration,
            max_utts=max_utts,
            trim=trim,
            bos_id=bos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            return_sample_id=return_sample_id,
            return_speaker_id=return_speaker_id,
            target_speakers=target_speakers,
            finetune_data_percentage=finetune_data_percentage,
        )
