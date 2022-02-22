import math

import torch
import torch.nn as nn
from nemo.collections.tts.modules.submodules import LinearNorm


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, scaled: bool, max_sequence_length: int = 5000):
        super().__init__()
        self.scaled = scaled

        if scaled:
            self.alpha = nn.Parameter(torch.ones(1))

        pos_enc = self.create_pos_encoding(hidden_size, max_sequence_length)
        self.register_buffer("pos_enc", pos_enc)

    def create_pos_encoding(self, hidden_size, max_sequence_length):
        pos_enc = torch.zeros(max_sequence_length, hidden_size)
        position = torch.arange(0.0, max_sequence_length).unsqueeze(1)
        coef = -math.log(10000.0) / hidden_size
        div_term = torch.exp(coef * torch.arange(0.0, hidden_size, 2))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc.div_(math.sqrt(hidden_size))

        return pos_enc

    def forward(self, inp):
        """Forward pass of positional encoding
        Args:
            inp (torch.Tensor): (batch, seq_len, hidden_size)
        """
        pos_enc = self.pos_enc[: inp.shape[1], :]
        if self.scaled:
            return inp + pos_enc.unsqueeze(0) * self.alpha

        return inp + pos_enc.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropattn, droplayer):
        super().__init__()

        self.n_head = n_head
        self.d_head = d_head
        self.d_model = d_model

        self.scale = d_head ** 0.5

        self.query = LinearNorm(d_model, d_head * n_head, w_init_gain="linear")
        self.key = LinearNorm(d_model, d_head * n_head, w_init_gain="linear")
        self.value = LinearNorm(d_model, d_head * n_head, w_init_gain="linear")
        self.output_layer = LinearNorm(d_head * n_head, d_model, w_init_gain="linear")

        self.dropattn = nn.Dropout(dropattn)
        self.droplayer = nn.Dropout(droplayer)

    def forward(self, q, k, v, mask):
        """Compute scaled dot-product attention
        Args:
            q (torch.Tensor): (batch, seq_len_q, d_model)
            k (torch.Tensor): (batch, seq_len_k, d_model)
            v (torch.Tensor): (batch, seq_len_k, d_model)
            mask (torch.Tensor): (batch, 1, seq_len_q, seq_len_k)
        """
        q, k, v = self.query(q), self.key(k), self.value(v)

        # (batch, seq_len, n_head, d_head)
        q = q.view(q.shape[0], q.shape[1], self.n_head, self.d_head)
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.d_head)
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.d_head)

        # (batch, n_head, seq_len, d_head)
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # (batch, n_head, seq_len_q, seq_len_k)
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn_score += mask.to(attn_score.dtype)

        attn_mask = torch.softmax(attn_score, dim=-1)
        attn = self.dropattn(attn_mask)

        # (batch, n_head, seq_len_q, d_model)
        x = torch.matmul(attn, v)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], -1, self.n_head * self.d_head)

        x = self.output_layer(x)
        x = self.droplayer(x)

        return x, attn_mask


class PositionWiseFF(nn.Module):
    def __init__(self, hidden_size, inner_size, ffn_dropout=0.0):
        super().__init__()

        self.dense_in = nn.Linear(hidden_size, inner_size)
        self.dense_out = nn.Linear(inner_size, hidden_size)
        self.layer_dropout = nn.Dropout(ffn_dropout)
        self.act_fn = nn.ReLU()

    def forward(self, hidden_states):
        output_states = self.dense_in(hidden_states)
        output_states = self.act_fn(output_states)
        output_states = self.dense_out(output_states)
        output_states = self.layer_dropout(output_states)

        return output_states


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        head_size: int,
        num_attention_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
        ffn_dropout: float,
        pre_ln: bool,
    ):
        super().__init__()
        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            num_attention_heads, hidden_size, head_size, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout)

    def forward_preln(self, encoder_x, encoder_mask):
        residual = encoder_x
        encoder_x = self.layer_norm_1(encoder_x)
        self_attn_output, attn_mask = self.first_sub_layer(encoder_x, encoder_x, encoder_x, encoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual

        return output_states, attn_mask

    def forward_postln(self, encoder_x, encoder_mask):
        residual = encoder_x
        self_attn_output, attn_mask = self.first_sub_layer(encoder_x, encoder_x, encoder_x, encoder_mask)
        self_attn_output += residual
        self_attn_output = self.layer_norm_1(self_attn_output)

        residual = self_attn_output
        output_states = self.second_sub_layer(self_attn_output)
        output_states += residual
        output_states = self.layer_norm_2(output_states)

        return output_states, attn_mask

    def forward(self, encoder_x, encoder_mask):
        if self.pre_ln:
            return self.forward_preln(encoder_x, encoder_mask)
        else:
            return self.forward_postln(encoder_x, encoder_mask)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        head_size: int,
        num_attention_heads: int,
        attn_score_dropout: float,
        attn_layer_dropout: float,
        ffn_dropout: float,
        pre_ln: bool,
    ):
        super().__init__()

        self.pre_ln = pre_ln
        self.layer_norm_1 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.first_sub_layer = MultiHeadAttention(
            num_attention_heads, hidden_size, head_size, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_2 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.second_sub_layer = MultiHeadAttention(
            num_attention_heads, hidden_size, head_size, attn_score_dropout, attn_layer_dropout
        )
        self.layer_norm_3 = nn.LayerNorm(hidden_size, eps=1e-5)
        self.third_sub_layer = PositionWiseFF(hidden_size, inner_size, ffn_dropout)

    def forward_preln(self, decoder_x, decoder_mask, encoder_x, encoder_mask):
        residual = decoder_x
        decoder_x = self.layer_norm_1(decoder_x)
        self_attn_output, decoder_attn = self.first_sub_layer(decoder_x, decoder_x, decoder_x, decoder_mask)
        self_attn_output += residual

        residual = self_attn_output
        self_attn_output = self.layer_norm_2(self_attn_output)
        enc_dec_attn_output, enc_dec_attn = self.second_sub_layer(self_attn_output, encoder_x, encoder_x, encoder_mask)
        enc_dec_attn_output += residual

        residual = enc_dec_attn_output
        enc_dec_attn_output = self.layer_norm_3(enc_dec_attn_output)
        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += residual

        return output_states, decoder_attn, enc_dec_attn

    def forward_postln(self, decoder_x, decoder_mask, encoder_x, encoder_mask):
        self_attn_output, decoder_attn = self.first_sub_layer(decoder_x, decoder_x, decoder_x, decoder_mask)
        self_attn_output += decoder_x
        self_attn_output = self.layer_norm_1(self_attn_output)

        enc_dec_attn_output, enc_dec_attn = self.second_sub_layer(self_attn_output, encoder_x, encoder_x, encoder_mask)
        enc_dec_attn_output += self_attn_output
        enc_dec_attn_output = self.layer_norm_2(enc_dec_attn_output)

        output_states = self.third_sub_layer(enc_dec_attn_output)
        output_states += enc_dec_attn_output
        output_states = self.layer_norm_3(output_states)

        return output_states, decoder_attn, enc_dec_attn

    def forward(self, decoder_x, decoder_mask, encoder_x, encoder_mask):
        if self.pre_ln:
            return self.forward_preln(decoder_x, decoder_mask, encoder_x, encoder_mask)
        else:
            return self.forward_postln(decoder_x, decoder_mask, encoder_x, encoder_mask)
