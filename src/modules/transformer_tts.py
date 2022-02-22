import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo.collections.tts.modules.submodules import ConvNorm, LinearNorm
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LengthsType,
    EncodedRepresentation,
    SequenceToSequenceAlignmentType,
    MelSpectrogramType,
    LogitsType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.utils import logging

from src.modules.transformer_submodules import PositionalEncoding
from src.modules.transformer import TransformerEncoder, TransformerDecoder


def parse_attention_outputs(attn_out, seq_lenghts):
    """Parse alignments from the last layer
    Args:
        self_attn_out (torch.Tensor): (layer, batch, heads, seq_len_q, seq_len_k)
        seq_lenghts (torch.Tensor): lengths of seq_len_q (batch,)
    Returns:
        masked attention plots
    """
    attn_out = attn_out[-1]
    mask = get_mask_from_lengths(seq_lenghts, max_len=attn_out.shape[-2])  # (batch, seq_len)
    mask = mask.unsqueeze(1).unsqueeze(-1)  # (batch, 1, seq_len_q, 1)
    attn_out = attn_out.masked_fill(~mask, 0.0)

    return attn_out


class Encoder(NeuralModule):
    def __init__(
        self,
        encoder_n_convolutions: int,
        encoder_embedding_dim: int,
        encoder_kernel_size: int,
        encoder_dropout_p: float,
        pos_dropout_p: float,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        head_size: int,
        num_attention_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
        ffn_dropout: float,
        pre_ln: bool,
        pre_ln_final_layer_norm: bool,
    ):
        super().__init__()

        convolutions = []
        for _ in range(encoder_n_convolutions):
            conv_layer = torch.nn.Sequential(
                ConvNorm(
                    encoder_embedding_dim,
                    encoder_embedding_dim,
                    kernel_size=encoder_kernel_size,
                    stride=1,
                    padding=int((encoder_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="relu",
                ),
                torch.nn.BatchNorm1d(encoder_embedding_dim),
            )
            convolutions.append(conv_layer)
        self.convolutions = torch.nn.ModuleList(convolutions)
        self.projection = LinearNorm(encoder_embedding_dim, hidden_size)

        self.encoder_dropout_p = encoder_dropout_p

        self.pos_encoding = PositionalEncoding(hidden_size, scaled=True)
        self.pos_dropout = nn.Dropout(p=pos_dropout_p)

        self.transformer_encoder = TransformerEncoder(
            num_layers,
            hidden_size,
            inner_size,
            head_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            pre_ln,
            pre_ln_final_layer_norm,
        )

    @property
    def input_types(self):
        return {
            "token_embedding": NeuralType(("B", "T", "D"), EmbeddedTextType()),
            "token_len": NeuralType(("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "encoder_embedding": NeuralType(("B", "T", "D"), EmbeddedTextType()),
            "cached_states": NeuralType(("D", "B", "T", "D"), EncodedRepresentation()),
            "parsed_enc_attn": NeuralType(("B", "H", "T", "T"), SequenceToSequenceAlignmentType()),
        }

    @typecheck()
    def forward(self, *, token_embedding, token_len):
        """
        Args:
            token_embedding (torch.Tensor): (batch, hidden_size, seq_len)
            token_len (torch.Tensor): (batch,)
        """
        token_embedding = token_embedding.transpose(1, 2)
        for conv in self.convolutions:
            token_embedding = F.dropout(F.relu(conv(token_embedding)), self.encoder_dropout_p, self.training)

        token_embedding = token_embedding.transpose(1, 2)
        token_embedding = self.projection(token_embedding)

        attn_input = self.pos_encoding(token_embedding)
        attn_input = self.pos_dropout(attn_input)

        token_mask = get_mask_from_lengths(token_len)
        encoder_embedding, cached_states, cached_attn = self.transformer_encoder(attn_input, token_mask)
        
        parsed_enc_attn = parse_attention_outputs(cached_attn, token_len)

        return encoder_embedding, cached_states, parsed_enc_attn


class Decoder(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        prenet_dropout_p: float,
        pos_dropout_p: float,
        num_layers: int,
        hidden_size: int,
        inner_size: int,
        head_size: int,
        num_attention_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
        ffn_dropout: float,
        pre_ln: bool,
        pre_ln_final_layer_norm: bool,
        gate_threshold: float,
        early_stopping: bool,
        max_decoder_steps: int,
    ):
        super().__init__()
        self.n_mel_channels = n_mel_channels

        # Prenet structure as in https://arxiv.org/pdf/1809.08895.pdf
        self.prenet = nn.Sequential(
            nn.Linear(n_mel_channels, hidden_size),
            nn.ReLU(),
            nn.Dropout(prenet_dropout_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.pos_encoding = PositionalEncoding(hidden_size, scaled=True)
        self.pos_dropout = nn.Dropout(p=pos_dropout_p)

        self.transformer_decoder = TransformerDecoder(
            num_layers,
            hidden_size,
            inner_size,
            head_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            pre_ln,
            pre_ln_final_layer_norm,
        )

        self.linear_projection = LinearNorm(hidden_size, n_mel_channels)

        self.gate_layer = LinearNorm(hidden_size, 1, w_init_gain="sigmoid")

        self.gate_threshold = gate_threshold
        self.early_stopping = early_stopping
        self.max_decoder_steps = max_decoder_steps

    @property
    def input_types(self):
        input_dict = {
            "encoder_out": NeuralType(("B", "T", "D"), EmbeddedTextType()),
            "encoder_len": NeuralType(("B"), LengthsType()),
        }
        if self.training:
            input_dict["decoder_input"] = NeuralType(("B", "T", "D"), MelSpectrogramType())
            input_dict["decoder_len"] = NeuralType(("B"), LengthsType())
        return input_dict

    @property
    def output_types(self):
        output_dict = {
            "mel_outputs": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "gate_outputs": NeuralType(("B", "T"), LogitsType()),
            "parsed_decoder_attn": NeuralType(("B", "H", "T", "T"), SequenceToSequenceAlignmentType()),
            "parsed_enc_dec_attn": NeuralType(("B", "H", "T", "T"), SequenceToSequenceAlignmentType()),
        }
        if self.training:
            output_dict["cached_states"] = NeuralType(("D", "B", "T", "D"), EncodedRepresentation())
        else:
            output_dict["mel_lengths"] = NeuralType(("B"), LengthsType())
        return output_dict

    @typecheck()
    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(**kwargs)
        return self.infer(**kwargs)

    def train_forward(self, *, encoder_out, encoder_len, decoder_input, decoder_len):
        go_frame = torch.zeros(encoder_out.shape[0], 1, self.n_mel_channels).type_as(encoder_out)
        decoder_input = torch.cat([go_frame, decoder_input], dim=1)
        # (batch, dec_seq_len + 1, n_mel_channels)
        decoder_input = self.prenet(decoder_input)

        decoder_input = self.pos_encoding(decoder_input)
        decoder_input = self.pos_dropout(decoder_input)

        decoder_mask = get_mask_from_lengths(decoder_len, max_len=decoder_input.shape[1])
        encoder_mask = get_mask_from_lengths(encoder_len, max_len=encoder_out.shape[1])
        decoder_states, cached_states, cached_decoder_attn, cached_enc_dec_attn = self.transformer_decoder(
            decoder_input, decoder_mask, encoder_out, encoder_mask
        )

        mel_outputs = self.linear_projection(decoder_states)
        gate_outputs = self.gate_layer(decoder_states).squeeze()

        parsed_decoder_attn = parse_attention_outputs(cached_decoder_attn, decoder_len)
        parsed_enc_dec_attn = parse_attention_outputs(cached_enc_dec_attn, decoder_len)

        return mel_outputs, gate_outputs, parsed_decoder_attn, parsed_enc_dec_attn, cached_states

    def infer(self, *, encoder_out, encoder_len):
        decoder_input = torch.zeros(encoder_out.shape[0], 1, self.n_mel_channels).type_as(encoder_out)

        mel_lengths = torch.zeros([encoder_out.shape[0]]).type_as(encoder_len)
        decoder_len = torch.ones([encoder_out.shape[0]]).type_as(encoder_len)
        not_finished = torch.ones([encoder_out.shape[0]]).type_as(encoder_len)

        mel_outputs, gate_outputs = [], []
        stepped = False

        encoder_mask = get_mask_from_lengths(encoder_len)
        while True:
            if len(mel_outputs) == 0:
                decoder_input = self.prenet(decoder_input)
            else:
                last_mel_frame = mel_outputs[-1]
                last_mel_frame = self.prenet(last_mel_frame)
                decoder_input = torch.cat([decoder_input, last_mel_frame], dim=1)

            pos_enc = self.pos_encoding(decoder_input)

            decoder_mask = get_mask_from_lengths(decoder_len)
            decoder_states, _, decoder_attn, enc_dec_attn = self.transformer_decoder(
                pos_enc, decoder_mask, encoder_out, encoder_mask
            )

            mel_out = self.linear_projection(decoder_states)
            gate_out = self.gate_layer(decoder_states)[:, -1, -1]

            dec = torch.le(torch.sigmoid(gate_out), self.gate_threshold).to(torch.int32)

            not_finished = not_finished * dec
            mel_lengths += not_finished
            decoder_len += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0 and stepped:
                break
            stepped = True

            mel_outputs += [mel_out[:, -1, :].unsqueeze(1)]
            gate_outputs += [gate_out.unsqueeze(1)]

            if len(mel_outputs) == self.max_decoder_steps:
                logging.warning("Reached max decoder steps %d.", self.max_decoder_steps)
                break

        mel_outputs = torch.cat(mel_outputs, dim=1)
        gate_outputs = torch.cat(gate_outputs, dim=1)
        parsed_decoder_attn = parse_attention_outputs(decoder_attn, decoder_len)
        parsed_enc_dec_attn = parse_attention_outputs(enc_dec_attn, decoder_len)

        return mel_outputs, gate_outputs, parsed_decoder_attn, parsed_enc_dec_attn, mel_lengths


class Postnet(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        postnet_embedding_dim: int,
        postnet_kernel_size: int,
        postnet_n_convolutions: int,
        p_dropout: float,
    ):
        super().__init__()
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    n_mel_channels,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="tanh",
                ),
                torch.nn.BatchNorm1d(postnet_embedding_dim),
            )
        )

        for _ in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                torch.nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                        w_init_gain="tanh",
                    ),
                    torch.nn.BatchNorm1d(postnet_embedding_dim),
                )
            )

        self.convolutions.append(
            torch.nn.Sequential(
                ConvNorm(
                    postnet_embedding_dim,
                    n_mel_channels,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                    w_init_gain="linear",
                ),
                torch.nn.BatchNorm1d(n_mel_channels),
            )
        )
        self.p_dropout = p_dropout

    @property
    def input_types(self):
        return {
            "mel_spec": NeuralType(("B", "T", "D"), MelSpectrogramType()),
        }

    @property
    def output_types(self):
        return {
            "mel_spec": NeuralType(("B", "T", "D"), MelSpectrogramType()),
        }

    @typecheck()
    def forward(self, *, mel_spec):
        mel_spec_out = mel_spec.transpose(1, 2)
        for i in range(len(self.convolutions) - 1):
            mel_spec_out = F.dropout(torch.tanh(self.convolutions[i](mel_spec_out)), self.p_dropout, self.training)
        mel_spec_out = F.dropout(self.convolutions[-1](mel_spec_out), self.p_dropout, self.training)

        return mel_spec + mel_spec_out.transpose(1, 2)
