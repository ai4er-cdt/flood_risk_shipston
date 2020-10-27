import os

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src import constants
from src.models import RunoffModel


def train_model(config):
    """
    Train model with PyTorch Lightning and log with Wandb.

    Explanation of unusual Trainer flags:
    accelerator=config.parallel_engine: Set PyTorch parallel execution engine to
    DistributedDataParallel by default, which is faster than DataParallel with
    > 2 GPUs or multiple nodes. If `config.cuda=False` then this is `None`.

    auto_select_gpus=config.cuda: Find the `gpus` most available GPUs to use,
    but only if we select `config.cuda=True` to allow GPU training.

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
    run_dir: str = os.path.join(constants.SAVE_PATH, config.run_name)
    # Force all runs to log to the herbie/shipston project and allow anonymous logging without a wandb account.
    wandb_logger = WandbLogger(name=config.run_name, save_dir=run_dir, entity='herbie', project='shipston',
                               save_code=False, anonymous=True)
    ckpt_path: str = os.path.join(run_dir, 'checkpoints', "{epoch}")
    ckpt = ModelCheckpoint(filepath=ckpt_path, period=config.mode.checkpoint_freq, monitor=config.mode.test_metric,
                           save_top_k=2, mode='max')

    # Instantiate Trainer
    trainer = Trainer(accelerator=config.parallel_engine, auto_select_gpus=config.cuda, gpus=config.gpus,
                      benchmark=True, deterministic=True, checkpoint_callback=ckpt, precision=config.precision,
                      prepare_data_per_node=False, max_epochs=config.mode.epochs, logger=wandb_logger,
                      log_every_n_steps=config.mode.log_steps)

    # Train model
    trainer.fit(runoff_model)

    # Load best checkpoint
    runoff_model = RunoffModel.load_from_checkpoint(ckpt.best_model_path)

    # Save weights from checkpoint
    statedict_path: str = os.path.join(run_dir, 'saved_models', f"{config.model.type}.pt")
    os.makedirs(os.path.dirname(statedict_path), exist_ok=True)
    torch.save(runoff_model.model.state_dict(), statedict_path)

    # Test and get Monte Carlo Dropout uncertainties.
    if config.mode.mc_dropout:
        for _ in range(config.mode.mc_dropout_iters):
            trainer.test(runoff_model)
        runoff_model.plot_results()
        config.mode.mc_dropout = False

    trainer.test(runoff_model)
    runoff_model.plot_results()
