from typing import Dict, Tuple

import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from src.models.lstm import LSTM_Model
from src.models.metrics import calc_nse
from src.preprocessing.camelsgb import CamelsGB

# TODO: implement optional wandb integration that doesn't activate unless wandb
# is installed? No, just mention in readme that you can do `wandb off`.


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


def train_model(config):
    device_str: str = config.gpu_ids[0]
    device: torch.device = torch.device(f"cuda:{device_str}" if config.cuda and torch.cuda.is_available() else "cpu")

    # Training data
    ds_train = CamelsGB(features=config.dataset.features,
                        seq_length=config.dataset.seq_length,
                        train_test_split=config.dataset.train_test_split,
                        train=True,
                        basin_ids=[config.dataset.basin_ids],
                        dates=config.mode.date_range)
    # Use `pin_memory=True` here for asynchronous data transfer to the GPU.
    train_loader = DataLoader(dataset=ds_train,
                           batch_size=config.mode.batch_size,
                           shuffle=config.dataset.shuffle,
                           num_workers=config.dataset.num_workers,
                           pin_memory=True)
    # Test data. We use the feature means/stds of the training period for normalization
    means: Dict[str, float] = ds_train.get_means()
    stds: Dict[str, float] = ds_train.get_stds()
    ds_test = CamelsGB(features=config.dataset.features,
                        seq_length=config.dataset.seq_length,
                        train_test_split=config.dataset.train_test_split,
                        train=False,
                        basin_ids=[config.dataset.basin_ids],
                        dates=config.mode.date_range,
                        means=means, stds=stds)
    test_loader = DataLoader(dataset=ds_test, batch_size=2048, shuffle=config.dataset.shuffle,
                        num_workers=config.dataset.num_workers, pin_memory=True)

    model = LSTM_Model(hidden_units=config.model.hidden_units,
                       num_features=5,
                       dropout_rate=config.model.dropout_rate,
                       num_layers=config.model.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.mode.learning_rate)
    loss_func = nn.MSELoss()

    for i in range(config.mode.epochs):
        train_epoch(model, optimizer, train_loader, loss_func, i + 1)
        obs, preds = eval_model(model, test_loader)
        preds = ds_test.local_rescale(preds.numpy(), variable='output')
        nse: float = calc_nse(obs.numpy(), preds)
        tqdm.tqdm.write(f"Validation NSE: {nse:.2f}")
