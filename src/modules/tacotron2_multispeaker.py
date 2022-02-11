from pathlib import Path
from typing import Optional

import torch
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.modules.submodules import Attention, LinearNorm, Prenet
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EmbeddedTextType,
    LabelsType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from torch.autograd import Variable
from torch.nn import functional as F


class Decoder(NeuralModule):
    def __init__(
        self,
        n_mel_channels: int,
        n_frames_per_step: int,
        encoder_embedding_dim: int,
        attention_dim: int,
        attention_location_n_filters: int,
        attention_location_kernel_size: int,
        attention_rnn_dim: int,
        decoder_rnn_dim: int,
        prenet_dim: int,
        speaker_embed_dim: int,
        pretrained_table_path: Optional[str],
        num_speakers: int,
        max_decoder_steps: int,
        gate_threshold: float,
        p_attention_dropout: float,
        p_decoder_dropout: float,
        early_stopping: bool,
        prenet_p_dropout: float = 0.5,
    ):
        """
        Modified Tacotron 2 Decoder. Using speaker embedding table.
        """
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.n_frames_per_step = n_frames_per_step
        self.encoder_embedding_dim = encoder_embedding_dim
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.speaker_embed_dim = speaker_embed_dim
        self.num_speakers = num_speakers
        self.max_decoder_steps = max_decoder_steps
        self.gate_threshold = gate_threshold
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout
        self.early_stopping = early_stopping

        if pretrained_table_path is not None and not Path(pretrained_table_path).exists():
            raise ValueError(f"No such file {pretrained_table_path}")

        if pretrained_table_path is None:
            self.spk_embedding = torch.nn.Embedding(self.num_speakers, self.speaker_embed_dim)
        else:
            d = torch.load(pretrained_table_path)
            self.spk_embedding = torch.nn.Embedding.from_pretrained(d["embeddings"])

        self.prenet = Prenet(n_mel_channels * n_frames_per_step, [prenet_dim, prenet_dim], prenet_p_dropout)

        self.attention_rnn = torch.nn.LSTMCell(
            prenet_dim + encoder_embedding_dim + speaker_embed_dim, attention_rnn_dim
        )

        self.attention_layer = Attention(
            attention_rnn_dim,
            encoder_embedding_dim + speaker_embed_dim,
            attention_dim,
            attention_location_n_filters,
            attention_location_kernel_size,
        )

        self.decoder_rnn = torch.nn.LSTMCell(
            attention_rnn_dim + encoder_embedding_dim + speaker_embed_dim, decoder_rnn_dim, 1
        )

        self.linear_projection = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim + speaker_embed_dim, n_mel_channels * n_frames_per_step
        )

        self.gate_layer = LinearNorm(
            decoder_rnn_dim + encoder_embedding_dim + speaker_embed_dim, 1, bias=True, w_init_gain="sigmoid"
        )

    @property
    def input_types(self):
        input_dict = {
            "memory": NeuralType(("B", "T", "D"), EmbeddedTextType()),
            "memory_lengths": NeuralType(("B"), LengthsType()),
            "speaker_idx": NeuralType(("B"), LabelsType()),
        }
        if self.training:
            input_dict["decoder_inputs"] = NeuralType(("B", "D", "T"), MelSpectrogramType())
        return input_dict

    @property
    def output_types(self):
        output_dict = {
            "mel_outputs": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "gate_outputs": NeuralType(("B", "T"), LogitsType()),
            "alignments": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
        }
        if not self.training:
            output_dict["mel_lengths"] = NeuralType(("B"), LengthsType())
        return output_dict

    @typecheck()
    def forward(self, *args, **kwargs):
        if self.training:
            return self.train_forward(**kwargs)
        return self.infer(**kwargs)

    def get_go_frame(self, memory):
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(B, MAX_TIME).zero_())
        self.attention_context = Variable(
            memory.data.new(B, self.encoder_embedding_dim + self.speaker_embed_dim).zero_()
        )

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1) / self.n_frames_per_step),
            -1,
        )
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        # (T_out, B) -> (B, T_out)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        # Add a -1 to prevent squeezing the batch dimension in case
        # batch is 1
        gate_outputs = torch.stack(gate_outputs).squeeze(-1).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step
        mel_outputs = mel_outputs.view(mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        cell_input = torch.cat((decoder_input, self.attention_context), -1)

        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell)
        )
        self.attention_hidden = F.dropout(self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)),
            dim=1,
        )
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden,
            self.memory,
            self.processed_memory,
            attention_weights_cat,
            self.mask,
        )

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), -1)

        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell)
        )
        self.decoder_hidden = F.dropout(self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat((self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def train_forward(self, *, memory, decoder_inputs, memory_lengths, speaker_idx):
        decoder_input = self.get_go_frame(memory).unsqueeze(0)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)
        decoder_inputs = self.prenet(decoder_inputs)

        speaker_embed = self.spk_embedding(speaker_idx).unsqueeze(1)  # batch x speaker_embed_dim
        speaker_embed = torch.repeat_interleave(speaker_embed, memory.size(1), 1)
        memory = torch.cat([memory, speaker_embed], dim=2)
        self.initialize_decoder_states(memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)
        return mel_outputs, gate_outputs, alignments

    def infer(self, *, memory, memory_lengths, speaker_idx):
        decoder_input = self.get_go_frame(memory)

        if memory.size(0) > 1:
            mask = ~get_mask_from_lengths(memory_lengths)
        else:
            mask = None

        speaker_embed = self.spk_embedding(speaker_idx).unsqueeze(1)
        speaker_embed = torch.repeat_interleave(speaker_embed, memory.size(1), 1)
        memory = torch.cat([memory, speaker_embed], dim=2)
        self.initialize_decoder_states(memory, mask=mask)

        mel_lengths = torch.zeros([memory.size(0)], dtype=torch.int32)
        not_finished = torch.ones([memory.size(0)], dtype=torch.int32)
        if torch.cuda.is_available():
            mel_lengths = mel_lengths.cuda()
            not_finished = not_finished.cuda()

        mel_outputs, gate_outputs, alignments = [], [], []
        stepped = False
        while True:
            decoder_input = self.prenet(decoder_input, inference=True)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            dec = torch.le(torch.sigmoid(gate_output.data), self.gate_threshold).to(torch.int32).squeeze(1)

            not_finished = not_finished * dec
            mel_lengths += not_finished

            if self.early_stopping and torch.sum(not_finished) == 0 and stepped:
                break
            stepped = True

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if len(mel_outputs) == self.max_decoder_steps:
                logging.warning("Reached max decoder steps %d.", self.max_decoder_steps)
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments, mel_lengths
