import pytorch_lightning as pl
from nemo.core.config import hydra_runner
from nemo.utils.exp_manager import exp_manager

from src.models.hifigan import HifiGanModel


@hydra_runner(config_path="conf/hifigan", config_name="hifigan_22050")
def main(cfg):
    trainer = pl.Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))
    model = HifiGanModel(cfg=cfg.model, trainer=trainer)
    trainer.fit(model)


if __name__ == "__main__":
    main()
