import random
from typing import Dict, Optional, Union

import torch
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment
from nemo.collections.common.parts.preprocessing import collections, parsers
from nemo.core.classes import Dataset
from nemo.core.neural_types.elements import *
from nemo.core.neural_types.neural_type import NeuralType


class AudioDataset(Dataset):
    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports."""
        return {
            "audio_signal": NeuralType(("B", "T"), AudioSignal()),
            "a_sig_length": NeuralType(tuple("B"), LengthsType()),
        }

    def __init__(
        self,
        manifest_filepath: Union[str, "pathlib.Path"],
        n_segments: int,
        sample_rate: int,
        max_duration: Optional[float] = None,
        min_duration: Optional[float] = None,
        trim: Optional[bool] = False,
        truncate_to: Optional[int] = 1,
    ):
        """
        Mostly compliant with nemo.collections.asr.data.datalayers.AudioToTextDataset except it only returns Audio
        without text. Dataset that loads tensors via a json file containing paths to audio files, transcripts, and
        durations (in seconds). Each new line is a different sample. Note that text is required, but is ignored for
        AudioDataset. Example below:
        {"audio_filepath": "/path/to/audio.wav", "text_filepath":
        "/path/to/audio.txt", "duration": 23.147}
        ...
        {"audio_filepath": "/path/to/audio.wav", "text": "the
        transcription", "offset": 301.75, "duration": 0.82, "utt":
        "utterance_id", "ctm_utt": "en_4156", "side": "A"}
        Args:
            manifest_filepath (str, Path): Path to manifest json as described above. Can be comma-separated paths
                such as "train_1.json,train_2.json" which is treated as two separate json files.
            n_segments (int): The length of audio in samples to load. For example, given a sample rate of 16kHz, and
                n_segments=16000, a random 1 second section of audio from the clip will be loaded. The section will
                be randomly sampled everytime the audio is batched. Can be set to -1 to load the entire audio.
            max_duration (float): If audio exceeds this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            min_duration(float): If audio is less than this length in seconds, it is filtered from the dataset.
                Defaults to None, which does not filter any audio.
            trim (bool): Whether to use librosa.effects.trim on the audio clip
            truncate_to (int): Ensures that the audio segment returned is a multiple of truncate_to.
                Defaults to 1, which does no truncating.
        """

        self.collection = collections.ASRAudioText(
            manifests_files=manifest_filepath.split(","),
            parser=parsers.make_parser(),
            min_duration=min_duration,
            max_duration=max_duration,
        )
        self.trim = trim
        self.n_segments = n_segments
        self.sample_rate = sample_rate
        self.truncate_to = truncate_to

    def _collate_fn(self, batch):
        """
        Takes a batch: a lists of length batch_size, defined in the dataloader. Returns 2 padded and batched
        tensors corresponding to the audio and audio_length.
        """

        def find_max_len(seq, index):
            max_len = -1
            for item in seq:
                if item[index].size(0) > max_len:
                    max_len = item[index].size(0)
            return max_len

        batch_size = len(batch)

        audio_signal, audio_lengths = None, None
        if batch[0][0] is not None:
            if self.n_segments > 0:
                max_audio_len = self.n_segments
            else:
                max_audio_len = find_max_len(batch, 0)

            audio_signal = torch.zeros(batch_size, max_audio_len, dtype=torch.float)
            audio_lengths = []
            for i, sample in enumerate(batch):
                audio_signal[i].narrow(0, 0, sample[0].size(0)).copy_(sample[0])
                audio_lengths.append(sample[1])
            audio_lengths = torch.tensor(audio_lengths, dtype=torch.long)

        return audio_signal, audio_lengths

    def __getitem__(self, index):
        """
        Given a index, returns audio and audio_length of the corresponding element. Audio clips of n_segments are
        randomly chosen if the audio is longer than n_segments.
        """
        example = self.collection[index]
        segment = AudioSegment.from_file(example.audio_file, target_sr=self.sample_rate, trim=self.trim)
        features = segment.samples
        if self.n_segments > 0 and features.shape[0] > self.n_segments:
            max_audio_start = features.shape[0] - self.n_segments
            audio_start = random.randint(0, max_audio_start)
            features = features[audio_start : audio_start + self.n_segments]

        features = torch.tensor(features)
        audio, audio_length = features, torch.tensor(features.shape[0]).long()

        truncate = audio_length % self.truncate_to
        if truncate != 0:
            audio_length -= truncate.long()
            audio = audio[:audio_length]

        return audio, audio_length

    def __len__(self):
        return len(self.collection)
