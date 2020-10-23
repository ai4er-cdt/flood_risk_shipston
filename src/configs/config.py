import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig, OmegaConf

from src import constants


@dataclass
class ModeConfig:
    train: bool = MISSING
    batch_size: int = MISSING
    date_range: List[str] = MISSING


@dataclass
class TrainConfig(ModeConfig):
    epochs: int = MISSING
    learning_rate: float = MISSING
    optimiser: str = MISSING
    loss_fn: str = MISSING
    checkpoint_freq: int = MISSING
    log_steps: int = MISSING
    fine_tune: bool = MISSING
    mc_dropout: bool = MISSING


@dataclass
class TestConfig(ModeConfig):
    pass


@dataclass
class ModelConfig:
    num_layers: int = MISSING
    type: str = MISSING


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
    basins_frac: float = MISSING
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
    run_name: str = MISSING


cs = ConfigStore.instance()
cs.store(name="config", node=ConfigClass)
cs.store(group="mode", name="train", node=TrainConfig)
cs.store(group="mode", name="test", node=TestConfig)
cs.store(group="model", name="lstm", node=LSTMConfig)


def validate_config(cfg: DictConfig) -> DictConfig:
    if cfg.run_name is None:
        raise TypeError("The `run_name` argument is mandatory.")
    # TODO: Implement dict schema for talking about features
    cfg.dataset.num_features = len(cfg.dataset.features)
    # TODO: Validate all string arguments
    # Make sure num_workers isn't too high.
    core_count = multiprocessing.cpu_count()
    if cfg.dataset.num_workers > core_count * 2:
        cfg.dataset.num_workers = core_count
    # Set data_dir
    if cfg.dataset.data_dir is None:
        cfg.dataset.data_dir = Path(constants.SRC_PATH) / 'data' / 'CAMELS-GB'

    print('----------------- Options ---------------')
    print(OmegaConf.to_yaml(cfg))
    print('----------------- End -------------------')
    return cfg


def validate_features(cfg: DictConfig, feature_list: List[str]) -> DictConfig:
    pass
