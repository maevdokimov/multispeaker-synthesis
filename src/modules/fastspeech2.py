import numpy as np
import torch
import torch.nn as nn
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.modules.fastspeech2_submodules import LengthRegulator, VariancePredictor
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import EncodedRepresentation, LengthsType, RegressionValuesType, TokenDurationType
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class VarianceAdaptor(NeuralModule):
    def __init__(
        self,
        d_model=256,
        dropout=0.2,
        dur_d_hidden=256,
        dur_kernel_size=3,
        pitch=True,
        log_pitch=True,
        n_f0_bins=256,
        pitch_kernel_size=3,
        pitch_min=80.0,
        pitch_max=800.0,
        energy=True,
        n_energy_bins=256,
        energy_kernel_size=3,
        energy_min=0.0,
        energy_max=600.0,
    ):
        """
        FastSpeech 2 variance adaptor, which adds information like duration, pitch, etc. to the phoneme encoding.
        Sets of conv1D blocks with ReLU and dropout.
        Args:
            d_model: Input and hidden dimension. Defaults to 256 (default encoder output dim).
            dropout: Variance adaptor dropout. Defaults to 0.2.
            dur_d_hidden: Hidden dim of the duration predictor. Defaults to 256.
            dur_kernel_size: Kernel size for the duration predictor. Defaults to 3.
            pitch (bool): Whether or not to use the pitch predictor.
            log_pitch (bool): If True, uses log pitch. Defaults to True.
            n_f0_bins: Number of F0 bins for the pitch predictor. Defaults to 256.
            pitch_kernel_size: Kernel size for the pitch predictor. Defaults to 3.
            pitch_min: Defaults to 80.0.
            pitch_max: Defaults to 800.0.
            pitch_d_hidden: Hidden dim of the pitch predictor.
            energy (bool): Whether or not to use the energy predictor.
            n_energy_bins: Number of energy bins. Defaults to 256.
            energy_kernel_size: Kernel size for the energy predictor. Defaults to 3.
            energy_min: Defaults to 0.0.
            energy_max: Defaults to 600.0.
        """
        super().__init__()

        # -- Duration Setup --
        self.duration_predictor = VariancePredictor(
            d_model=d_model, d_inner=dur_d_hidden, kernel_size=dur_kernel_size, dropout=dropout
        )
        self.length_regulator = LengthRegulator()

        self.pitch = pitch
        self.energy = energy

        # -- Pitch Setup --
        # NOTE: Pitch is clamped to 1e-5 which gets mapped to bin 1. But it is padded with 0s that get mapped to bin 0.
        if self.pitch:
            if log_pitch:
                pitch_min = np.log(pitch_min)
                pitch_max = np.log(pitch_max)
            pitch_operator = torch.exp if log_pitch else lambda x: x
            pitch_bins = pitch_operator(torch.linspace(start=pitch_min, end=pitch_max, steps=n_f0_bins - 1))
            # Prepend 0 for unvoiced frames
            pitch_bins = torch.cat((torch.tensor([0.0]), pitch_bins))

            self.register_buffer("pitch_bins", pitch_bins)
            self.pitch_predictor = VariancePredictor(
                d_model=d_model, d_inner=n_f0_bins, kernel_size=pitch_kernel_size, dropout=dropout
            )
            # Predictor outputs values directly rather than one-hot vectors, therefore Embedding
            self.pitch_lookup = nn.Embedding(n_f0_bins, d_model)

        # -- Energy Setup --
        if self.energy:
            self.register_buffer(  # Linear scale bins
                "energy_bins", torch.linspace(start=energy_min, end=energy_max, steps=n_energy_bins - 1)
            )
            self.energy_predictor = VariancePredictor(
                d_model=d_model,
                d_inner=n_energy_bins,
                kernel_size=energy_kernel_size,
                dropout=dropout,
            )
            self.energy_lookup = nn.Embedding(n_energy_bins, d_model)

    @property
    def input_types(self):
        return {
            "x": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "x_len": NeuralType(("B"), LengthsType()),
            "dur_target": NeuralType(("B", "T"), TokenDurationType(), optional=True),
            "pitch_target": NeuralType(("B", "T"), RegressionValuesType(), optional=True),
            "energy_target": NeuralType(("B", "T"), RegressionValuesType(), optional=True),
            "spec_len": NeuralType(("B"), LengthsType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(("B", "T", "D"), EncodedRepresentation()),
            "log_dur_preds": NeuralType(("B", "T"), TokenDurationType()),
            "pitch_preds": NeuralType(("B", "T"), RegressionValuesType()),
            "energy_preds": NeuralType(("B", "T"), RegressionValuesType()),
            "spec_len": NeuralType(("B"), LengthsType()),
        }

    @typecheck()
    def forward(self, *, x, x_len, dur_target=None, pitch_target=None, energy_target=None, spec_len=None):
        """
        Args:
            x: Input from the encoder.
            x_len: Length of the input.
            dur_target:  Duration targets for the duration predictor. Needs to be passed in during training.
            pitch_target: Pitch targets for the pitch predictor. Needs to be passed in during training.
            energy_target: Energy targets for the energy predictor. Needs to be passed in during training.
            spec_len: Target spectrogram length. Needs to be passed in during training.
        """
        # Duration predictions (or ground truth) fed into Length Regulator to
        # expand the hidden states of the encoder embedding
        log_dur_preds = self.duration_predictor(x)
        log_dur_preds.masked_fill_(~get_mask_from_lengths(x_len), 0)
        # Output is Batch, Time
        if dur_target is not None:
            dur_out = self.length_regulator(x, dur_target)
        else:
            dur_preds = torch.clamp_min(torch.round(torch.exp(log_dur_preds)) - 1, 0).long()
            if not torch.sum(dur_preds, dim=1).bool().all():
                logging.error("Duration prediction failed on this batch. Settings to 1s")
                dur_preds += 1
            dur_out = self.length_regulator(x, dur_preds)
            spec_len = torch.sum(dur_preds, dim=1)
        out = dur_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        # Pitch
        pitch_preds = None
        if self.pitch:
            # Possible future work:
            #   Add pitch spectrogram prediction & conversion back to pitch contour using iCWT
            #   (see Appendix C of the FastSpeech 2/2s paper).
            pitch_preds = self.pitch_predictor(dur_out)
            pitch_preds.masked_fill_(~get_mask_from_lengths(spec_len), 0)
            if pitch_target is not None:
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_target, self.pitch_bins))
            else:
                pitch_out = self.pitch_lookup(torch.bucketize(pitch_preds.detach(), self.pitch_bins))
            out = out + pitch_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        # Energy
        energy_preds = None
        if self.energy:
            energy_preds = self.energy_predictor(dur_out)
            if energy_target is not None:
                energy_out = self.energy_lookup(torch.bucketize(energy_target, self.energy_bins))
            else:
                energy_out = self.energy_lookup(torch.bucketize(energy_preds.detach(), self.energy_bins))
            out = out + energy_out
        out *= get_mask_from_lengths(spec_len).unsqueeze(-1)

        return out, log_dur_preds, pitch_preds, energy_preds, spec_len
