import torch
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LossType, SequenceToSequenceAlignmentType
from nemo.core.neural_types.neural_type import NeuralType


class GuidedAttentionLoss(Loss):
    def __init__(self, g: float, scale: float):
        """Guided attention loss https://arxiv.org/pdf/1710.08969.pdf
        Args:
            g (float): attention weight scale
            scale (float): loss scale
        """
        super().__init__()

        self.g = g
        self.scale = scale

    @property
    def input_types(self):
        return {
            "alignments": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
            "spec_target_len": NeuralType(("B"), LengthsType()),
            "encoder_target_len": NeuralType(("B"), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, *, alignments, spec_target_len, encoder_target_len):
        _losses = []

        for i, align_map in enumerate(torch.chunk(alignments, alignments.shape[0], dim=0)):
            T, N = spec_target_len[i], encoder_target_len[i]
            align_map = align_map.squeeze()[:T, :N]  # (t_seq_len, enc_seq_len)

            _arr_n = torch.arange(N).type_as(alignments).repeat(T, 1)
            _arr_t = torch.arange(T).type_as(alignments).repeat(N, 1).transpose(0, 1)
            W = 1 - torch.exp(-((_arr_n / N - _arr_t / T) ** 2) / 2 / self.g ** 2)

            _losses.append(torch.mean(align_map[:T, :N] * W))

        return sum(_losses) * self.scale / len(_losses)
