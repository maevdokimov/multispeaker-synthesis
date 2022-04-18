from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from hydra.utils import instantiate
from nemo.collections.common.parts.preprocessing import parsers
from nemo.collections.tts.helpers.helpers import get_mask_from_lengths
from nemo.collections.tts.losses.tacotron2loss import Tacotron2Loss
from nemo.collections.tts.models.base import SpectrogramGenerator
from nemo.core.classes.common import typecheck
from nemo.core.neural_types.elements import (
    AudioSignal,
    EmbeddedTextType,
    LengthsType,
    LogitsType,
    MelSpectrogramType,
    NormalDistributionLogVarianceType,
    NormalDistributionMeanType,
    SequenceToSequenceAlignmentType,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import logging
from omegaconf import MISSING, DictConfig, OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger
from torch import nn

from src.utils.helpers import tacotron2_log_to_tb_func


@dataclass
class Preprocessor:
    _target_: str = MISSING
    pad_value: float = MISSING


@dataclass
class Tacotron2Config:
    preprocessor: Preprocessor = Preprocessor()
    encoder: Dict[Any, Any] = MISSING
    decoder: Dict[Any, Any] = MISSING
    postnet: Dict[Any, Any] = MISSING
    reference_encoder: Dict[Any, Any] = MISSING
    labels: List = MISSING
    train_ds: Optional[Dict[Any, Any]] = None
    validation_ds: Optional[Dict[Any, Any]] = None


class Tacotron2Model(SpectrogramGenerator):
    """Tacotron 2 Model that is used to generate mel spectrograms from text"""

    def __init__(self, cfg: DictConfig, trainer: "Trainer" = None):
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        super().__init__(cfg=cfg, trainer=trainer)

        schema = OmegaConf.structured(Tacotron2Config)
        # ModelPT ensures that cfg is a DictConfig, but do this second check in case ModelPT changes
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        elif not isinstance(cfg, DictConfig):
            raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
        # Ensure passed cfg is compliant with schema
        try:
            OmegaConf.merge(cfg, schema)
            self.pad_value = self._cfg.preprocessor.pad_value
        except ConfigAttributeError:
            self.pad_value = self._cfg.preprocessor.params.pad_value
            logging.warning(
                "Your config is using an old NeMo yaml configuration. Please ensure that the yaml matches the "
                "current version in the main branch for future compatibility."
            )

        self._parser = None
        self.audio_to_melspec_precessor = instantiate(self._cfg.preprocessor)
        self.text_embedding = nn.Embedding(len(cfg.labels) + 3, 512)
        self.encoder = instantiate(self._cfg.encoder)
        self.decoder = instantiate(self._cfg.decoder)
        self.postnet = instantiate(self._cfg.postnet)
        self.reference_encoder = instantiate(self._cfg.reference_encoder)

        self.calculate_loss = True
        self.calculate_guided_loss = False

        self.loss = Tacotron2Loss()
        self.vae_loss = instantiate(self._cfg.vae_loss)
        if "guided_attention_loss" in self._cfg.keys():
            self.guided_attention_loss = instantiate(self._cfg.guided_attention_loss)
            self.calculate_guided_loss = True

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
            labels=self._cfg.labels,
            name=name,
            unk_id=unk_id,
            blank_id=blank_id,
            do_normalize=do_normalize,
        )
        return self._parser

    def parse(self, str_input: str) -> torch.tensor:
        tokens = self.parser(str_input)
        # Parser doesn't add bos and eos ids, so maunally add it
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
                "ref_audio": NeuralType(("B", "T"), AudioSignal(), optional=True),
                "ref_audio_len": NeuralType(("B"), LengthsType(), optional=True),
            }
        else:
            return {
                "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
                "token_len": NeuralType(("B"), LengthsType()),
                "audio": NeuralType(("B", "T"), AudioSignal(), optional=True),
                "audio_len": NeuralType(("B"), LengthsType(), optional=True),
                "ref_audio": NeuralType(("B", "T"), AudioSignal(), optional=True),
                "ref_audio_len": NeuralType(("B"), LengthsType(), optional=True),
            }

    @property
    def output_types(self):
        if not self.calculate_loss and not self.training:
            return {
                "spec_pred_dec": NeuralType(("B", "D", "T"), MelSpectrogramType()),
                "spec_pred_postnet": NeuralType(("B", "D", "T"), MelSpectrogramType()),
                "gate_pred": NeuralType(("B", "T"), LogitsType()),
                "alignments": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
                "pred_length": NeuralType(("B"), LengthsType()),
            }
        return {
            "spec_pred_dec": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "spec_pred_postnet": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "gate_pred": NeuralType(("B", "T"), LogitsType()),
            "spec_target": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "spec_target_len": NeuralType(("B"), LengthsType()),
            "ref_spec": NeuralType(("B", "D", "T"), MelSpectrogramType()),
            "ref_spec_len": NeuralType(("B"), LengthsType()),
            "alignments": NeuralType(("B", "T", "T"), SequenceToSequenceAlignmentType()),
            "mu": NeuralType(("B", "D"), NormalDistributionMeanType()),
            "logvar": NeuralType(("B", "D"), NormalDistributionLogVarianceType()),
        }

    @typecheck()
    def forward(self, *, tokens, token_len, audio=None, audio_len=None, ref_audio=None, ref_audio_len=None):
        if audio is not None and audio_len is not None:
            spec_target, spec_target_len = self.audio_to_melspec_precessor(audio, audio_len)
            ref_spec, ref_spec_len = spec_target, spec_target_len
        if ref_audio is not None and ref_audio_len is not None:
            ref_spec, ref_spec_len = self.audio_to_melspec_precessor(ref_audio, ref_audio_len)

        token_embedding = self.text_embedding(tokens).transpose(1, 2)
        encoder_embedding = self.encoder(token_embedding=token_embedding, token_len=token_len)
        reference_embedding, mu, logvar, _ = self.reference_encoder(ref_spec=ref_spec)
        reference_embedding = reference_embedding.unsqueeze(1).expand_as(encoder_embedding)

        if self.training:
            spec_pred_dec, gate_pred, alignments = self.decoder(
                memory=encoder_embedding + reference_embedding, decoder_inputs=spec_target, memory_lengths=token_len
            )
        else:
            spec_pred_dec, gate_pred, alignments, pred_length = self.decoder(
                memory=encoder_embedding + reference_embedding, memory_lengths=token_len
            )
        spec_pred_postnet = self.postnet(mel_spec=spec_pred_dec)

        if not self.calculate_loss:
            return spec_pred_dec, spec_pred_postnet, gate_pred, alignments, pred_length
        return (
            spec_pred_dec,
            spec_pred_postnet,
            gate_pred,
            spec_target,
            spec_target_len,
            ref_spec,
            ref_spec_len,
            alignments,
            mu,
            logvar,
        )

    @typecheck(
        input_types={
            "tokens": NeuralType(("B", "T"), EmbeddedTextType()),
            "ref_audio": NeuralType(("B", "T"), AudioSignal()),
        },
        output_types={"spec": NeuralType(("B", "D", "T"), MelSpectrogramType())},
    )
    def generate_spectrogram(self, *, tokens, ref_audio):
        self.eval()
        self.calculate_loss = False
        token_len = torch.tensor([len(i) for i in tokens]).to(self.device)
        ref_audio_len = torch.tensor([len(i) for i in ref_audio]).to(self.device)
        tensors = self(tokens=tokens, token_len=token_len, ref_audio=ref_audio, ref_audio_len=ref_audio_len)
        spectrogram_pred = tensors[1]

        if spectrogram_pred.shape[0] > 1:
            # Silence all frames past the predicted end
            mask = ~get_mask_from_lengths(tensors[-1])
            mask = mask.expand(spectrogram_pred.shape[1], mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)
            spectrogram_pred.data.masked_fill_(mask, self.pad_value)

        return spectrogram_pred

    def training_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        (
            spec_pred_dec,
            spec_pred_postnet,
            gate_pred,
            spec_target,
            spec_target_len,
            _,
            _,
            alignments,
            mu,
            logvar,
        ) = self.forward(audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len)

        tacotron_loss, _ = self.loss(
            spec_pred_dec=spec_pred_dec,
            spec_pred_postnet=spec_pred_postnet,
            gate_pred=gate_pred,
            spec_target=spec_target,
            spec_target_len=spec_target_len,
            pad_value=self.pad_value,
        )

        _kl_loss = self.vae_loss(mu=mu, logvar=logvar)
        kl_weight = self.vae_loss.kl_anneal_function(self.global_step)
        kl_loss = kl_weight * _kl_loss

        if self.calculate_guided_loss:
            guided_loss = self.guided_attention_loss(
                alignments=alignments,
                spec_target_len=spec_target_len,
                encoder_target_len=token_len,
            )
            loss = tacotron_loss + guided_loss + kl_loss
            log = {
                "loss": loss,
                "kl_loss": kl_loss,
                "tacotron_loss": tacotron_loss,
                "guided_loss": guided_loss,
                "kl_weight": kl_weight,
            }
        else:
            loss = tacotron_loss + kl_loss
            log = {"loss": loss, "tacotron_loss": tacotron_loss, "kl_loss": kl_loss, "kl_weight": kl_weight}

        output = {
            "loss": loss,
            "progress_bar": {"training_loss": loss},
            "log": log,
        }

        return output

    def validation_step(self, batch, batch_idx):
        audio, audio_len, tokens, token_len = batch
        # spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, alignments = self.forward(
        # spec_pred_dec, spec_pred_postnet, gate_pred, spec_target, spec_target_len, _, _, alignments, _, _ = self.forward(
        (
            spec_pred_dec,
            spec_pred_postnet,
            gate_pred,
            spec_target,
            spec_target_len,
            _,
            _,
            alignments,
            _,
            _,
        ) = self.forward(audio=audio, audio_len=audio_len, tokens=tokens, token_len=token_len)

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
