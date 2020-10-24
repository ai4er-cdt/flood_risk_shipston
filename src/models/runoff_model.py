from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from src.models import LSTM_Model, calc_nse
from src.preprocessing import CamelsGB
from torch.utils.data import DataLoader


class RunoffModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        # Long term - think about using hydra.utils.instantiate here.
        self.model = LSTM_Model(hidden_units=self.config.model.hidden_units,
                                num_features=self.config.dataset.num_features,
                                dropout_rate=self.config.model.dropout_rate,
                                num_layers=self.config.model.num_layers)
        self.loss = torch.nn.MSELoss()

    def forward(self, batch):
        x, y = batch
        return self.model(x)

    def training_step(self, batch):
        x, y = batch
        y_hat = self(x)  # Calls self.forward(x)
        loss = self.loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.mode.learning_rate)
        return opt

    def validation_step(self, batch):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

    def validation_epoch_end(self, outputs: List) -> None:
        ys_list = []
        preds_list = []
        for (y, y_hat) in outputs:
            ys_list.append(y)
            preds_list.append(y_hat)
        preds = self.train_set.local_rescale(torch.cat(preds_list), variable='output')
        nse: float = calc_nse(torch.cat(ys_list), preds)
        self.log('NSE', nse)

    def test_step(self, batch):
        return self.validation_step(batch)

    def test_epoch_end(self, outputs: List) -> None:
        ys_list = []
        preds_list = []
        for (y, y_hat) in outputs:
            ys_list.append(y)
            preds_list.append(y_hat)
        ys = torch.cat(ys_list)
        preds = self.train_set.local_rescale(torch.cat(preds_list), variable='output')
        nse: float = calc_nse(ys, preds)
        self.log('NSE', nse)

        # Plot results
        start_date = self.test_set.dates[0]
        end_date = self.test_set.dates[1] + pd.DateOffset(days=1)
        date_range = pd.date_range(start_date, end_date)
        fig, ax = plt.subplots(figsize=(16, 8))
        # Play around with alpha here to see uncertainty better!
        ax.plot(date_range, ys, label="observation", alpha=0.8)
        # Replace dropout_preds_mean with preds if dropout section skipped
        if self.config.mode.mc_dropout:
            # TODO: implement full MC dropout in the model.
            raise NotImplementedError
        else:
            ax.plot(date_range, preds, label="prediction")

        # ax.plot(date_range, dropout_preds_mean, label="prediction")
        # ax.fill_between(date_range, dropout_preds_mean - np.sqrt(dropout_preds_var), dropout_preds_mean +
        # np.sqrt(dropout_preds_var), alpha=0.5,color='orange', #label = 'pred uncertainty')
        # Hash out this line if dropout section skipped
        ax.legend()
        ax.set_title(f"Test set NSE: {nse:.3f}")
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (mm/d)")
        # Save to save_dir and log to wandb.

    def prepare_data(self) -> None:
        # TODO: Download data here.
        raise NotImplementedError

    def setup(self, stage: str):
        # `stage` can be either `fit` or `test`.
        self.train_set = CamelsGB(data_dir=self.config.dataset.data_dir,
                                    features=self.config.dataset.features,
                                    basins_frac=self.config.dataset.basins_frac,
                                    dates=self.config.mode.date_range,
                                    train=True,
                                    seq_length=self.config.dataset.seq_length,
                                    train_test_split=self.config.dataset.train_test_split)
        self.means: Dict[str, float] = self.train_set.get_means()
        self.stds: Dict[str, float] = self.train_set.get_stds()
        self.test_set = CamelsGB(data_dir=self.config.dataset.data_dir,
                                 features=self.config.dataset.features,
                                 basins_frac=self.config.dataset.basins_frac,
                                 dates=self.config.mode.date_range,
                                 train=False,
                                 seq_length=self.config.dataset.seq_length,
                                 train_test_split=self.config.dataset.train_test_split,
                                 means=self.means, stds=self.stds)

    def train_dataloader(self):
        # Use `pin_memory=True` here for asynchronous data transfer to the GPU.
        dataloader = DataLoader(dataset=self.train_set,
                                batch_size=self.config.mode.batch_size,
                                shuffle=self.config.dataset.shuffle,
                                num_workers=self.config.dataset.num_workers,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(dataset=self.test_set, batch_size=2048, shuffle=self.config.dataset.shuffle,
                                num_workers=self.config.dataset.num_workers, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
