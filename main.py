import os

import hydra
from omegaconf import DictConfig, OmegaConf

from src import config, constants, train_model


@hydra.main(config_path=os.path.join(constants.SRC_PATH, 'configs'), config_name="config")
def main(cfg: DictConfig) -> None:
    cfg = config.validate_config(cfg)
    print(OmegaConf.to_yaml(cfg))

    if cfg.mode.train:
        train_model(cfg)


if __name__ == "__main__":
    main()
