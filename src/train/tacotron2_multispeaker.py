import pytorch_lightning as pl
import torch
from nemo.collections.common.callbacks import LogEpochTimeCallback
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from src.models.tacotron2_multispeaker import Tacotron2Multispeaker


@hydra_runner(config_path="conf", config_name="tacotron2_multispeaker_22050_finetune")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    model = Tacotron2Multispeaker(cfg=cfg.model, trainer=trainer)
    if cfg.checkpoint_path is not None:
        d = torch.load(cfg.checkpoint_path)
        model.load_state_dict(d["state_dict"])

    lr_logger = pl.callbacks.LearningRateMonitor()
    epoch_time_logger = LogEpochTimeCallback()
    device_monitor = pl.callbacks.gpu_stats_monitor.GPUStatsMonitor()
    trainer.callbacks.extend([lr_logger, epoch_time_logger, device_monitor])

    trainer.fit(model)


if __name__ == "__main__":
    main()  # noqa pylint: disable=no-value-for-parameter
