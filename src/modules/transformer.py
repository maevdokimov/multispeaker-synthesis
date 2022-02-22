import copy

import torch
import torch.nn as nn
from nemo.collections.common.parts import form_attention_mask

from src.modules.transformer_submodules import TransformerEncoderLayer, TransformerDecoderLayer


class TransformerEncoder(nn.Module):
    def __init__(
        self,
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

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = TransformerEncoderLayer(
            hidden_size,
            inner_size,
            head_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            pre_ln,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, encoder_states, encoder_mask):
        """
        Args:
            encoder_states: output of the embedding_layer (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """
        encoder_attn_mask = form_attention_mask(encoder_mask)
        cached_states, cached_attn = [], []

        for layer in self.layers:
            encoder_states, attn_mask = layer(encoder_states, encoder_attn_mask)
            cached_states.append(encoder_states)
            cached_attn.append(attn_mask)

        if self.final_layer_norm is not None:
            encoder_states = self.final_layer_norm(encoder_states)

        return encoder_states, torch.stack(cached_states, dim=0), torch.stack(cached_attn, dim=0)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
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

        if pre_ln and pre_ln_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        else:
            self.final_layer_norm = None

        layer = TransformerDecoderLayer(
            hidden_size,
            inner_size,
            head_size,
            num_attention_heads,
            attn_score_dropout,
            attn_layer_dropout,
            ffn_dropout,
            pre_ln,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.diagonal = 0

    def forward(self, decoder_states, decoder_mask, encoder_states, encoder_mask):
        """
        Args:
            decoder_states: output of the embedding layer (B x L_dec x H)
            decoder_mask: decoder inputs mask (B x L_dec)
            encoder_states: output of the encoder (B x L_enc x H)
            encoder_mask: encoder inputs mask (B x L_enc)
        """

        decoder_attn_mask = form_attention_mask(decoder_mask, diagonal=self.diagonal)
        encoder_attn_mask = form_attention_mask(encoder_mask)
        cached_states, cached_decoder_attn, cached_enc_dec_attn = [], [], []

        for layer in self.layers:
            decoder_states, decoder_attn, enc_dec_attn = layer(
                decoder_states, decoder_attn_mask, encoder_states, encoder_attn_mask
            )
            cached_states.append(decoder_states)
            cached_decoder_attn.append(decoder_attn)
            cached_enc_dec_attn.append(enc_dec_attn)

        if self.final_layer_norm is not None:
            decoder_states = self.final_layer_norm(decoder_states)

        return (
            decoder_states,
            torch.stack(cached_states, dim=0),
            torch.stack(cached_decoder_attn, dim=0),
            torch.stack(cached_enc_dec_attn, dim=0),
        )
