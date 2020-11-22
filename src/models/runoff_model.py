import logging
import os
import shutil
import zipfile
from typing import List

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import requests
import torch
import wandb
from PIL import Image
from src.constants import *
from src.models import LSTMModel, calc_nse
from src.preprocessing import BaseDataset, CamelsGB, ShipstonDataset
from torch.utils.data import DataLoader


class RunoffModel(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        # Long term - think about using hydra.utils.instantiate here.
        self.model = LSTMModel(hidden_units=self.config.model.hidden_units,
                                num_features=self.config.dataset.num_features,
                                dropout_rate=self.config.model.dropout_rate,
                                num_layers=self.config.model.num_layers)
        self.loss = torch.nn.MSELoss()
        self.printer = logging.getLogger("lightning")

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config.mode.learning_rate)
        return opt

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)  # Calls self.forward(x)
        loss = self.loss(y_hat, y)
        self.log("Loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y, y_hat

    def validation_epoch_end(self, outputs: List) -> None:
        ys_list = []
        preds_list = []
        for (y, y_hat) in outputs:
            ys_list.append(y)
            preds_list.append(y_hat)
        preds = self.test_set.rescale(torch.cat(preds_list), input=False)
        test_metric: float = calc_nse(torch.cat(ys_list).cpu().numpy(), preds.cpu().numpy())
        self.log(self.config.mode.test_metric, test_metric)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_start(self):
        if self.config.mode.mc_dropout:
            self.model.train()

    def test_epoch_end(self, outputs: List) -> None:
        ys_list = []
        preds_list = []
        for (y, y_hat) in outputs:
            ys_list.append(y)
            preds_list.append(y_hat)
        preds = self.test_set.rescale(torch.cat(preds_list), input=False).cpu()
        self.ys = torch.cat(ys_list).cpu()

        if self.config.mode.mc_dropout:
            self.preds: torch.Tensor = torch.cat((self.preds, preds), dim=1) if hasattr(self, 'preds') else preds
        else:
            self.preds = preds
            self.test_metric: float = calc_nse(self.ys.numpy(), self.preds.numpy())
            self.log(f"Test {self.config.mode.test_metric}", self.test_metric)

    def plot_results(self) -> None:
        fig, ax = plt.subplots(figsize=(16, 8))
        x_axis: List[int] = list(range(len(self.ys)))
        # Play around with alpha here to see uncertainty better!
        ax.plot(x_axis, self.ys, label="observation", alpha=0.8)
        ax.plot(x_axis, self.preds, label="prediction")
        if self.config.mode.mc_dropout:
            preds_var, preds_mean = torch.var_mean(self.preds, dim=1)
            ax.fill_between(x_axis, preds_mean - torch.sqrt(preds_var), preds_mean +
                            torch.sqrt(preds_var), alpha=0.5, color='orange', label='pred uncertainty')
            plot_name: str = "Results-MCDropout"
            ax.set_title("MC Dropout Results")
        else:
            plot_name = "Results"
            ax.set_title(f"Test set NSE: {self.test_metric:.3f}")
        ax.legend()
        ax.set_xlabel("Day")
        ax.set_ylabel("Discharge (mm/d)")
        # Save plot to png file.
        fig.savefig(os.path.join(SAVE_PATH, self.config.run_name, f"{plot_name.lower()}.png"))
        # Convert plot to PIL image and log to wandb.
        pil_image = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        self.logger.experiment.log({plot_name: wandb.Image(pil_image, mode='RGB', caption=plot_name)})
        self.printer.info("Plot generated and saved.")

    def prepare_data(self):
        """Download the relevant dataset from a Dropbox link and extract it."""
        if self.config.dataset.type == 'shipston':
            if not os.path.exists(os.path.join(DATA_PATH, SHIPSTON_ID)):
                self.printer.info("Downloading data...")
                req = requests.get(SHIPSTON_URL, allow_redirects=True)
                open(os.path.join(DATA_PATH, SHIPSTON_ID), 'wb').write(req.content)
                self.printer.info("Data downloaded.")
        else:
            camels_dir: str = os.path.join(DATA_PATH, 'CAMELS-GB')
            # Check if data exists already.
            if not os.path.exists(camels_dir) or not os.path.isdir(camels_dir) or not os.listdir(camels_dir):
                self.printer.info("Downloading data...")
                write_path: str = os.path.join(DATA_PATH, f"{CAMELS_ID}.zip")
                # Download from Dropbox URL.
                req = requests.get(CAMELS_URL, stream=True)
                with open(write_path, 'wb') as file:
                    for chunk in req.iter_content(chunk_size=128):
                        file.write(chunk)
                # Extract zip file.
                with zipfile.ZipFile(write_path, 'r') as zip_ref:
                    zip_ref.extractall(DATA_PATH)
                # Move contents of inner directory to CAMELS-GB directory.
                source_dir: str = os.path.join(DATA_PATH, CAMELS_ID, 'data')
                os.mkdir(camels_dir)
                file_names: List[str] = os.listdir(source_dir)
                for file_name in file_names:
                    shutil.move(os.path.join(source_dir, file_name), os.path.join(camels_dir, file_name))
                # Delete zip file and waste folder.
                shutil.rmtree(os.path.join(DATA_PATH, CAMELS_ID))
                os.remove(write_path)
                self.printer.info("Data downloaded.")

    def setup(self, stage: str):
        # `stage` can be either "fit" or "test".
        if stage == 'fit':
            self.printer.info("Loading training set.")
            if self.config.dataset.type == 'shipston':
                self.train_set: BaseDataset = ShipstonDataset(features=self.config.dataset.features,
                                                              dates=self.config.mode.date_range,
                                                              train=True,
                                                              seq_length=self.config.dataset.seq_length,
                                                              train_test_split=self.config.dataset.train_test_split,
                                                              precision=self.config.precision)
            elif self.config.dataset.type == 'CAMELS-GB':
                self.train_set = CamelsGB(features=self.config.dataset.features,
                                          basins_frac=self.config.dataset.basins_frac,
                                          dates=self.config.mode.date_range,
                                          train=True,
                                          seq_length=self.config.dataset.seq_length,
                                          train_test_split=self.config.dataset.train_test_split,
                                          precision=self.config.precision)
            self.printer.info("Loaded training set.")
        self.printer.info("Loading test set.")
        if self.config.dataset.type == 'shipston':
            self.test_set: BaseDataset = ShipstonDataset(features=self.config.dataset.features,
                                                         dates=self.config.mode.date_range,
                                                         train=False,
                                                         seq_length=self.config.dataset.seq_length,
                                                         train_test_split=self.config.dataset.train_test_split,
                                                         precision=self.config.precision)
        else:
            self.test_set = CamelsGB(features=self.config.dataset.features,
                                     basins_frac=self.config.dataset.basins_frac,
                                     dates=self.config.mode.date_range,
                                     train=False,
                                     seq_length=self.config.dataset.seq_length,
                                     train_test_split=self.config.dataset.train_test_split,
                                     precision=self.config.precision)
        self.printer.info("Loaded test set.")

    def train_dataloader(self):
        # Use `pin_memory=True` here for asynchronous data transfer to the GPU, speeding up data loading.
        dataloader = DataLoader(dataset=self.train_set,
                                batch_size=self.config.mode.batch_size,
                                shuffle=self.config.dataset.shuffle,
                                num_workers=self.config.dataset.num_workers,
                                pin_memory=True)
        return dataloader

    def val_dataloader(self):
        # Best practice to use shuffle=False for validation and testing.
        dataloader = DataLoader(dataset=self.test_set, batch_size=2048, shuffle=False,
                                num_workers=self.config.dataset.num_workers, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
