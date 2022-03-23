import logging

import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from vapo.affordance.affordance_model import AffordanceModel
from vapo.affordance.dataloader.datasets import get_loaders


def print_cfg(cfg):
    print_cfg = OmegaConf.to_container(cfg)
    print_cfg.pop("dataset")
    print_cfg.pop("trainer")
    return OmegaConf.create(print_cfg)


@hydra.main(config_path="../config", config_name="cfg_affordance")
def train(cfg):
    print("Running configuration: ", cfg)
    logger = logging.getLogger(__name__)
    logger.info("Running configuration: %s", OmegaConf.to_yaml(print_cfg(cfg)))

    # Data split
    train_loader, val_loader, im_shape, _ = get_loaders(logger, cfg.dataset, cfg.dataloader)

    # 24hr format
    model_name = cfg.model_name

    # Initialize model
    checkpoint_loss_callback = ModelCheckpoint(
        monitor="val_total_loss",
        dirpath="trained_models",
        filename="%s_{epoch:02d}-{val_total_loss:.3f}" % model_name,
        save_top_k=2,
        verbose=True,
        mode="min",
    )

    checkpoint_miou_callback = ModelCheckpoint(
        monitor="val_miou",
        dirpath="trained_models",
        filename="%s_{epoch:02d}-{val_miou:.3f}" % model_name,
        save_top_k=2,
        verbose=True,
        mode="max",
        save_last=True,
    )

    wandb_logger = WandbLogger(name=model_name, **cfg.wandb)
    aff_model = AffordanceModel(
        cfg.model_cfg, n_classes=cfg.model_cfg.n_classes, input_channels=im_shape[0], cmd_log=logger
    )
    trainer = pl.Trainer(
        callbacks=[checkpoint_miou_callback, checkpoint_loss_callback], logger=wandb_logger, **cfg.trainer
    )
    trainer.fit(aff_model, train_loader, val_loader)


if __name__ == "__main__":
    train()
