from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import nn

from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    EmbeddedTextType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging

from src.utils.helpers import transformer_tts_log_to_tb_func
from src.losses.transformer_tts_loss import TransformerTTSLoss


@dataclass
class Preprocessor:
    _target_: str = MISSING
    pad_value: float = MISSING


@dataclass
class TransformerTTSConfig:
    preprocessor: Preprocessor = Preprocessor()
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    postnet: Dict[Any, Any] = MISSING
    labels: List = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class TransformerTTSModel(SpectrogramGenerator):
    def __init__(self, cfg: DictConfig, trainer: "Trainer" = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(TransformerTTSConfig)

        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")

        OmegaConf.merge(cfg, schema)
        self.pad_value = self._cfg.preprocessor.pad_value
        self.gate_target_weight = self._cfg.gate_target_weight

        self._parser = None
        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.text_embedding = nn.Embedding(len(cfg.labels) + 3, 512)
        self.encoder = instantiate(self._cfg.encoder)
        self.decoder = instantiate(self._cfg.decoder)
        self.postnet = instantiate(self._cfg.postnet)
        self.loss = TransformerTTSLoss()
        self.calculate_loss = True

    @property
    def parser(self):
        if self._parser is not None:
            return self._parser
        if self._validation_dl is not None:
            return self._validation_dl.dataset.manifest_processor.parser
        if self._test_dl is not None:
            return self._test_dl.dataset.manifest_processor.parser
        if self._train_dl is not None:
            return self._train_dl.dataset.manifest_processor.parser

        # Else construct a parser
        # Try to get params from validation, test, and then train
        params = {}
        try:
            params = self._cfg.validation_ds.dataset
        except ConfigAttributeError:
            pass
        if params == {}:
            try:
                params = self._cfg.test_ds.dataset
            except ConfigAttributeError:
                pass
        if params == {}:
            try:
                params = self._cfg.train_ds.dataset
            except ConfigAttributeError:
                pass

        name = params.get("parser", None) or "en"
        unk_id = params.get("unk_index", None) or -1
        blank_id = params.get("blank_index", None) or -1
        do_normalize = params.get("normalize", True)
        self._parser = parsers.make_parser(
            labels=self._cfg.labels, name=name, unk_id=unk_id, blank_id=blank_id, do_normalize=do_normalize,
        )
        return self._parser

    def parse(self, str_input: str) -> torch.tensor:
        tokens = self.parser(str_input)
        tokens = [len(self._cfg.labels)] + tokens + [len(self._cfg.labels) + 1]
        tokens_tensor = torch.tensor(tokens).unsqueeze_(0).to(self.device)

        return tokens_tensor

    @property
    def input_types(self):
        if self.training:
            return {
                "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
                "token_len": NeuralType(("B"), LengthsType()),
                "audio": NeuralType(("B", "T"), AudioSignal()),
                "audio_len": NeuralType(("B"), LengthsType()),
            }
        else:
            return {
                "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
                "token_len": NeuralType(("B"), LengthsType()),
                "audio": NeuralType(("B", "T"), AudioSignal(), optional=True),
                "audio_len": NeuralType(("B"), LengthsType(), optional=True),
            }

    @property
    def output_types(self):
        if not self.calculate_loss and not self.training:
            return {
                "spec_pred_dec": NeuralType(("B", "T", "D"), MelSpectrogramType()),
                "spec_pred_postnet": NeuralType(("B", "T", "D"), MelSpectrogramType()),
                "gate_pred": NeuralType(("B", "T"), LogitsType()),
                "pred_length": NeuralType(("B"), LengthsType()),
            }
        return {
            "spec_pred_dec": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "spec_pred_postnet": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "gate_pred": NeuralType(("B", "T"), LogitsType()),
            "spec_target": NeuralType(("B", "T", "D"), MelSpectrogramType()),
            "spec_target_len": NeuralType(("B"), LengthsType()),
            "alignments": NeuralType(("B", "H", "T", "T"), SequenceToSequenceAlignmentType()),
        }

    @typecheck()
    def forward(self, *, tokens, token_len, audio=None, audio_len=None):
        if audio is not None and audio_len is not None:
            spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
            spec_target = spec_target.transpose(1, 2)
        token_embedding = self.text_embedding(tokens)
        encoder_embedding, _, _ = self.encoder(token_embedding=token_embedding, token_len=token_len)
        if self.training:
            spec_pred_dec, gate_pred, _, alignments, _ = self.decoder(
                encoder_out=encoder_embedding,
                encoder_len=token_len,
                decoder_input=spec_target,
                decoder_len=spec_target_len,
            )
        else:
            spec_pred_dec, gate_pred, _, alignments, pred_length = self.decoder(
                encoder_out=encoder_embedding, encoder_len=token_len
            )

        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, pred_length
        return spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, _ = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss, _ = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
            gate_target_weight=self.gate_target_weight,
        )

        output = {
            "loss": loss,
            "progress_bar": {"training_loss": loss},
            "log": {"loss": loss},
        }
        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments = self.forward(
            audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len
        )

        loss, gate_target = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
            gate_target_weight=self.gate_target_weight,
        )
        return {
            "val_loss": loss,
            "mel_target": spec_target,
            "mel_postnet": spec_pred_postnet,
            "gate": gate_pred,
            "gate_target": gate_target,
            "alignments": alignments,
        }

    def generate_spectrogram(self, tokens: "torch.tensor", **kwargs) -> "torch.tensor":
        return super().generate_spectrogram(tokens, **kwargs)

    def __setup_dataloader_from_config(self, cfg, shuffle_should_be: bool = True, name: str = "train"):
        if "dataset" not in cfg or not isinstance(cfg.dataset, DictConfig):
            raise ValueError(f"No dataset for {name}")
        if "dataloader_params" not in cfg or not isinstance(cfg.dataloader_params, DictConfig):
            raise ValueError(f"No dataloder_params for {name}")
        if shuffle_should_be:
            if "shuffle" not in cfg.dataloader_params:
                logging.warning(
                    f"Shuffle should be set to True for {self}'s {name} dataloader but was not found in its "
                    "config. Manually setting to True"
                )
                with open_dict(cfg.dataloader_params):
                    cfg.dataloader_params.shuffle = True
            elif not cfg.dataloader_params.shuffle:
                logging.error(f"The {name} dataloader for {self} has shuffle set to False!!!")
        elif not shuffle_should_be and cfg.dataloader_params.shuffle:
            logging.error(f"The {name} dataloader for {self} has shuffle set to True!!!")

        labels = self._cfg.labels

        dataset = instantiate(
            cfg.dataset, labels=labels, bos_id=len(labels), eos_id=len(labels) + 1, pad_id=len(labels) + 2
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **cfg.dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self.__setup_dataloader_from_config(cfg)

    def setup_validation_data(self, cfg):
        self._validation_dl = self.__setup_dataloader_from_config(cfg, shuffle_should_be=False, name="validation")

    def validation_epoch_end(self, outputs):
        if self.logger is not None and self.logger.experiment is not None:
            tb_logger = self.logger.experiment
            if isinstance(self.logger, LoggerCollection):
                for logger in self.logger:
                    if isinstance(logger, TensorBoardLogger):
                        tb_logger = logger.experiment
                        break

            if self.global_step != 0:
                transformer_tts_log_to_tb_func(
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
