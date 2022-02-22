import torch
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.models.tacotron2 import Tacotron2Model
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import AudioSignal, EmbeddedTextType, LabelsType, LengthsType, MelSpectrogramType
from nemo.core.neural_types.neural_type import NeuralType
from omegaconf import DictConfig
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from src.utils.helpers import tacotron2_log_to_tb_func


class Tacotron2Multispeaker(Tacotron2Model):
    def __init__(self, cfg: DictConfig, trainer: "Trainer" = None):
        super().__init__(cfg, trainer)

    @property
    def input_types(self):
        if self.training:
            return {
                "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
                "token_len": NeuralType(("B"), LengthsType()),
                "speaker_idx": NeuralType(("B"), LabelsType()),
                "audio": NeuralType(("B", "T"), AudioSignal()),
                "audio_len": NeuralType(("B"), LengthsType()),
            }
        else:
            return {
                "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
                "token_len": NeuralType(("B"), LengthsType()),
                "speaker_idx": NeuralType(("B"), LabelsType()),
                "audio": NeuralType(("B", "T"), AudioSignal(), optional=True),
                "audio_len": NeuralType(("B"), LengthsType(), optional=True),
            }

    @typecheck()
    def forward(self, *, tokens, token_len, speaker_idx, audio=None, audio_len=None):
        if audio is not None and audio_len is not None:
            spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        if self.training:
            spec_pred_dec, gate_pred, alignments = self.decoder(
                memory=encoder_embedding, decoder_inputs=spec_target, memory_lengths=token_len, speaker_idx=speaker_idx
            )
        else:
            spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
                memory=encoder_embedding, memory_lengths=token_len, speaker_idx=speaker_idx
            )
        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

    @typecheck(
        input_types={
            "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
            "speaker_idx": NeuralType(("B"), LabelsType()),
        },
        output_types={"spec": NeuralType(("B", "D", "T"), MelSpectrogramType())},
    )
    def generate_spectrogram(self, *, tokens, speaker_idx):
        self.eval()
        self.calculate_loss = False
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        tensors = self(tokens=tokens, token_len=token_len, speaker_idx=speaker_idx)
        spectrogram_pred = tensors[1]

        if spectrogram_pred.shape[0] > 1:
            # Silence all frames past the predicted end
            mask = ~get_mask_from_lengths(tensors[-1])
            mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            spectrogram_pred.data.masked_fill_(mask, self.pad_value)

        return spectrogram_pred

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len, speaker_idx = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, _ = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len, speaker_idx=speaker_idx
        )

        loss, _ = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )

        output = {
            "loss": loss,
            "progress_bar": {"training_loss": loss},
            "log": {"loss": loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len, speaker_idx = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len, speaker_idx=speaker_idx
        )

        loss, gate_target = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )
        return {
            "val_loss": loss,
            "mel_target": spec_target,
            "mel_postnet": spec_pred_postnet,
            "gate": gate_pred,
            "gate_target": gate_target,
            "alignments": alignments,
        }

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break

            if self.global_step != 0:
                tacotron2_log_to_tb_func(
                    tb_logger,
                    outputs[0].values(),
                    self.global_step,
                    tag="val",
                    log_images=True,
                    add_audio=True,
                    sr=self.cfg.preprocessor.sample_rate,
                    n_fft=self.cfg.preprocessor.n_fft,
                    n_mels=self.cfg.preprocessor.nfilt,
                    fmax=self.cfg.preprocessor.highfreq or self.cfg.preprocessor.sample_rate // 2,
                    sample_idx=self.cfg.log_val_sample_idx or 0,
                )

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log("val_loss", avg_loss)
