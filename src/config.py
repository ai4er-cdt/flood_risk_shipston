import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

from src import constants


@dataclass
class ModeConfig:
    train: bool = MISSING
    batch_size: int = MISSING
    date_range: Optional[List[str]] = MISSING


@dataclass
class TrainConfig(ModeConfig):
    epochs: int = MISSING
    learning_rate: float = MISSING
    optimiser: str = MISSING
    loss_fn: str = MISSING
    checkpoint_freq: int = MISSING
    log_steps: int = MISSING
    fine_tune: bool = MISSING


@dataclass
class TestConfig(ModeConfig):
    pass


@dataclass
class ModelConfig:
    num_layers: int = MISSING


@dataclass
class LSTMConfig(ModelConfig):
    bidirectional: bool = MISSING
    hidden_units: int = MISSING
    dropout_rate: float = MISSING


@dataclass
class DatasetConfig:
    data_dir: Optional[str] = MISSING
    features: List[str] = MISSING
    seq_length: int = MISSING
    train_test_split: str = MISSING
    basin_ids: str = MISSING
    shuffle: bool = MISSING
    num_workers: int = MISSING


@dataclass
class ConfigClass:
    mode: ModeConfig = MISSING
    model: ModelConfig = MISSING
    dataset: DatasetConfig = DatasetConfig()
    cuda: bool = MISSING
    gpu_ids: List[int] = MISSING
    seed: int = MISSING
    save_dir: Optional[str] = MISSING
    run_name: str = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigClass)
cs.store(group="mode", name="train", node=TrainConfig)
cs.store(group="mode", name="test", node=TestConfig)
cs.store(group="model", name="lstm", node=LSTMConfig)


def validate_config(cfg: DictConfig) -> DictConfig:
    if cfg.run_name is None:
        raise TypeError("The `run_name` argument is mandatory.")
    # TODO: Decide on schema for talking about features
    cfg.dataset.features = list(cfg.dataset.features)
    # TODO: Validate all string arguments
    core_count = multiprocessing.cpu_count()
    if cfg.dataset.num_workers > core_count * 2:
        cfg.dataset.num_workers = core_count
    # Set save_dir and data_dir
    if cfg.dataset.data_dir is None:
        cfg.dataset.data_dir = Path(constants.SRC_PATH) / 'data' / 'CAMELS-GB'
    if cfg.save_dir is None:
        cfg.save_dir = Path(constants.PROJECT_PATH) / 'log'
        os.makedirs(cfg.save_dir, exist_ok=True)
    return cfg


def validate_features(cfg: DictConfig, feature_list: List[str]) -> DictConfig:
    pass
