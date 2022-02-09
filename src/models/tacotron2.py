import torch
from nemo.collections.tts.models.tacotron2 import Tacotron2Model
from pytorch_lightning.loggers import LoggerCollection, TensorBoardLogger

from src.utils.helpers import tacotron2_log_to_tb_func


class Tacotron2Model(Tacotron2Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
                    sample_idx=self.cfg.log_val_sample_idx or 0,
                )

        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()  # This reduces across batches, not workers!
        self.log("val_loss", avg_loss)
