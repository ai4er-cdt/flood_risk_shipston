import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src import constants
from src.models import RunoffModel


def train_model(config):
    # Set random seeds.
    seed_everything(config.seed)
    # TODO: Get GPU IDs that exist.
    runoff_model = RunoffModel(config)

    # Trainer
    wandb_dir = os.path.join(constants.SAVE_PATH, config.run_name)
    wandb_logger = WandbLogger(name=config.run_name, save_dir=wandb_dir, project='shipston', config=config)
    ckpt_path = os.path.join(constants.SAVE_PATH, config.run_name, 'checkpoints')
    ckpt = ModelCheckpoint(dirpath=ckpt_path, period=config.mode.checkpoint_freq)
    lr_logger = LearningRateMonitor()  # TODO: Test logging_interval='epoch'
    # TODO: Finalise trainer flags.
    trainer = Trainer(callbacks=[lr_logger], gpus=config.gpu_ids, max_epochs=config.epochs, checkpoint_callback=ckpt,
                      deterministic=True, logger=wandb_logger, log_every_n_steps=config.mode.log_steps)
    trainer.fit(runoff_model)

    # Load best checkpoint
    runoff_model = RunoffModel.load_from_checkpoint(ckpt.best_model_path)

    # Save weights from checkpoint
    statedict_path = os.path.join(constants.SAVE_PATH, config.run_name, 'saved_models', f"{config.model.type}.pt")
    os.makedirs(os.path.dirname(statedict_path), exist_ok=True)
    torch.save(runoff_model.model.state_dict(), statedict_path)
