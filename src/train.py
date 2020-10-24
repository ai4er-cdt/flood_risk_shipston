import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src import constants
from src.models import RunoffModel


def train_model(config):
    """
    Train model with PyTorch Lightning and log with Wandb.

    Explanation of unusual Trainer flags:
    accelerator='ddp': Set PyTorch parallel execution engine to
    DistributedDataParallel" which is faster than DataParallel with > 2 GPUs or
    multiple nodes.
    auto_select_gpus=True: Find the `gpus` most available GPUs to use.
    benchmark=True: Enable cuDNN optimisation algorithms (speeds up training
    when model input size is constant).
    deterministic=True: Forces model output to be deterministic given the same
    random seed.
    prepare_data_per_node=False: Calls the RunoffModel.prepare_data() hook (to
    download the dataset) only on one node, since we are using a cluster with a
    shared filesystem.
    """
    # Set random seeds.
    seed_everything(config.seed)

    runoff_model = RunoffModel(config)

    # Setup logging and checkpointing.
    # TODO: Set wandb dir
    wandb_dir = os.path.join(constants.SAVE_PATH, config.run_name)
    wandb_logger = WandbLogger(name=config.run_name, save_dir=wandb_dir, project='shipston', config=config)
    ckpt_path = os.path.join(constants.SAVE_PATH, config.run_name, 'checkpoints')
    # TODO: Try monitor='nse' here.
    ckpt = ModelCheckpoint(filepath=os.path.join(ckpt_path, "{epoch}"), period=config.mode.checkpoint_freq)
    lr_logger = LearningRateMonitor()  # TODO: Test logging_interval='epoch'

    # Instantiate Trainer
    trainer = Trainer(accelerator='ddp', auto_select_gpus=True, gpus=config.gpus, benchmark=True, deterministic=True,
                      callbacks=[lr_logger], checkpoint_callback=ckpt, prepare_data_per_node=False,
                      max_epochs=config.mode.epochs, logger=wandb_logger, log_every_n_steps=config.mode.log_steps)

    # Train model
    trainer.fit(runoff_model)

    # Load best checkpoint
    runoff_model = RunoffModel.load_from_checkpoint(ckpt.best_model_path)

    # Save weights from checkpoint
    statedict_path = os.path.join(constants.SAVE_PATH, config.run_name, 'saved_models', f"{config.model.type}.pt")
    os.makedirs(os.path.dirname(statedict_path), exist_ok=True)
    torch.save(runoff_model.model.state_dict(), statedict_path)
