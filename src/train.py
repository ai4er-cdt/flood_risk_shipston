from typing import Dict, Tuple

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from models.lstm import LSTM_Model
from models.metrics import calc_nse
from preprocessing.camelsgb import CamelsGB

# TODO: implement optional wandb integration that doesn't activate unless wandb
# is installed?
# TODO: options.py with argparse


def train_epoch(model: torch.nn.Module, optimizer: torch.optim.Optimizer, loader: torch.utils.data.DataLoader,
                loss_func: torch.nn.Module, epoch: int) -> None:
    """
    Train model for a single epoch.

    Note that we use `non_blocking=True` when transferring data to the device.
    This only does anything when using a GPU and when `pin_memory=True` is used
    in the DataLoader, in which case the data is transferred asynchronously to
    the GPU from pinned memory on the CPU, increasing GPU utilisation.

    Args:
        model (torch.nn.Module): A torch.nn.Module implementing the LSTM model.
        optimizer (torch.optim.Optimizer): A PyTorch optimizer class.
        loader (torch.utils.data.DataLoader): A PyTorch DataLoader with the
        training data.
        loss_func (torch.nn.Module): The loss function to minimize.
        epoch (int): The current epoch, used for the progress bar.
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(device, non_blocking=True), ys.to(device, non_blocking=True)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")


def eval_model(model: torch.nn.Module, loader: torch.utils.data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate the model.

    Args:
        model (torch.nn.Module): A torch.nn.Module implementing the LSTM model.
        loader (torch.utils.data.DataLoader): A PyTorch DataLoader with the data
        to evaluate.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Two Tensors containing the
        observations and predictions respectively for all minibatches in the
        data loader.
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(device, non_blocking=True)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)

    return torch.cat(obs), torch.cat(preds)


if __name__ == "__main__":
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hidden_size: int = 10  # Number of LSTM cells
    dropout_rate: float = 0.4  # Dropout rate of the final fully connected Layer [0.0, 1.0] - used for regularisation
    # and as a way to extract uncertainty (MC dropout)
    learning_rate: float = 3e-3
    sequence_length: int = 365  # Length of the meteorological record provided to the network
    batch_size = 256  # 512 is the value used in the paper

    # TODO: Pass list of features as an argument with sensible default, have num_features calculated from this
    # and passed to LSTM_Model for input.

    # Training data
    start_date: pd.Timestamp = pd.to_datetime("2000-12-09", format="%Y-%m-%d")
    end_date: pd.Timestamp = pd.to_datetime("2008-12-08", format="%Y-%m-%d")
    ds_train = CamelsGB(seq_length=sequence_length, mode="train", dates=[start_date, end_date])
    # Use `pin_memory=True` here for asynchronous data transfer to the GPU.
    tr_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, pin_memory=True)

    # Validation data. We use the feature means/stds of the training period for normalization
    means: Dict[str, float] = ds_train.get_means()
    stds: Dict[str, float] = ds_train.get_stds()
    start_date = pd.to_datetime("2008-12-09", format="%Y-%m-%d")
    end_date = pd.to_datetime("2010-12-08", format="%Y-%m-%d")
    ds_val = CamelsGB(seq_length=sequence_length, mode="eval", dates=[start_date, end_date],
                        means=means, stds=stds)
    val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False, pin_memory=True)

    # Test data. We use the feature means/stds of the training period for normalization
    start_date = pd.to_datetime("2010-12-09", format="%Y-%m-%d")
    end_date = pd.to_datetime("2015-09-29", format="%Y-%m-%d")
    ds_test = CamelsGB(seq_length=sequence_length, mode="eval", dates=[start_date, end_date],
                        means=means, stds=stds)
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False, pin_memory=True)

    model = LSTM_Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()

    n_epochs = 50

    for i in range(n_epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, i + 1)
        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.numpy(), variable='output')
        nse: float = calc_nse(obs.numpy(), preds)
        tqdm.tqdm.write(f"Validation NSE: {nse:.2f}")
