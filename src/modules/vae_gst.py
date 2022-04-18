import numpy as np
import torch
import torch.nn as nn
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.modules.fastspeech2_submodules import LengthRegulator, VariancePredictor
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    LabelsType,
    LengthsType,
    RegressionValuesType,
    TokenDurationType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging


class ReferenceEncoder(NeuralModule):
    def __init__(
        self,
        ref_enc_filters,
        output_dim,
        n_mels,
        kernel_size=3,
        stride=2,
        padding=1,
        z_latent_dim=32,
        ref_enc_gru_size=256,
    ):
        super().__init__()
        filters = [1] + ref_enc_filters

        convs = nn.ModuleList([])
        for i in range(len(ref_enc_filters) - 2):
            convs.append(
                nn.Conv2d(filters[i], filters[i + 1], kernel_size=kernel_size, stride=stride, padding=padding)
            )
            convs.append(nn.BatchNorm2d(filters[i + 1]))
            convs.append(nn.ReLU())

        self.convs = nn.Sequential(convs)

        out_channels = self.calculate_channels(n_mels, kernel_size, stride, padding, len(ref_enc_filters))
        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1] * out_channels, hidden_size=ref_enc_gru_size, batch_first=True
        )
        self.n_mels = n_mels

        self.out_mu = nn.Linear(ref_enc_gru_size, z_latent_dim)
        self.out_var = nn.Linear(ref_enc_gru_size, z_latent_dim)
        self.out_layer = nn.Linear(z_latent_dim, output_dim)

    def forward(self, inputs):
        out = inputs.unsqueeze(1)
        out = self.convs(out)

        out = out.transpose(1, 2)
        out = out.reshape(out.shape[0], out.shape[1], -1)

        _, out = self.gru(out)
        out = out.squeeze(0)

        mu = self.out_mu(out)
        logvar = self.out_var(out)
        z = self.reparameterize(mu, logvar)
        style_embed = self.out_layer(z)

        return style_embed, mu, logvar, z

    def calculate_channels(self, n_mels, kernel_size, stride, padding, n_convs):
        for _ in range(n_convs):
            n_mels = (n_mels - kernel_size + 2 * padding) // stride + 1

        return n_mels

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
