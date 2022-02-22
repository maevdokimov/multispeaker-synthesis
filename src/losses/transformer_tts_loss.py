import torch

from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import LengthsType, LogitsType, LossType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType


class TransformerTTSLoss(Loss):
    @property
    def input_types(self):
        return {
            "spec_pred_dec": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "spec_pred_postnet": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "gate_pred": NeuralType(("B", "T"), LogitsType()),
            "spec_target": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "spec_target_len": NeuralType(("B"), LengthsType()),
            "pad_value": NeuralType(),
            "gate_target_weight": NeuralType(),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
            "gate_target": NeuralType(("B", "T"), LogitsType()),  # Used for evaluation
        }

    @typecheck()
    def forward(
        self,
        *,
        spec_pred_dec,
        spec_pred_postnet,
        gate_pred,
        spec_target,
        spec_target_len,
        pad_value,
        gate_target_weight
    ):
        max_len = spec_target.shape[1]
        gate_target = torch.zeros(spec_target_len.shape[0], max_len)
        gate_target = gate_target.type_as(gate_pred)
        for i, length in enumerate(spec_target_len):
            gate_target[i, length.data - 1 :] = 1

        spec_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)

        if max_len < spec_pred_dec.shape[1]:
            spec_pred_dec = spec_pred_dec.narrow(1, 0, max_len)
            spec_pred_postnet = spec_pred_postnet.narrow(1, 0, max_len)
            gate_pred = gate_pred.narrow(1, 0, max_len).contiguous()
        elif max_len > spec_pred_dec.shape[1]:
            pad_amount = max_len - spec_pred_dec.shape[1]
            spec_pred_dec = torch.nn.functional.pad(spec_pred_dec, (0, 0, 0, pad_amount), value=pad_value)
            spec_pred_postnet = torch.nn.functional.pad(spec_pred_postnet, (0, 0, 0, pad_amount), value=pad_value)
            gate_pred = torch.nn.functional.pad(gate_pred, (0, pad_amount), value=1e3)
            max_len = spec_pred_dec.shape[1]

        mask = ~get_mask_from_lengths(spec_target_len, max_len=max_len)
        mask = mask.expand(spec_target.shape[2], mask.size(0), mask.size(1))
        mask = mask.permute(1, 2, 0)
        spec_pred_dec.data.masked_fill_(mask, pad_value)
        spec_pred_postnet.data.masked_fill_(mask, pad_value)
        gate_pred.data.masked_fill_(mask[:, :, 0], 1e3)

        gate_pred = gate_pred.view(-1, 1)
        rnn_mel_loss = torch.nn.functional.mse_loss(spec_pred_dec, spec_target)
        postnet_mel_loss = torch.nn.functional.mse_loss(spec_pred_postnet, spec_target)
        gate_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            gate_pred, gate_target, pos_weight=torch.tensor([gate_target_weight]).type_as(gate_pred)
        )
        return rnn_mel_loss + postnet_mel_loss + gate_loss, gate_target
