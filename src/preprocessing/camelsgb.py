import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# TODO: iterate through all basin files and find out if any flow data has nans in the middle of it rather than just
# at the start (or perhaps the end?)
# TODO: combine two basins into one and figure out how to split train/test, probably add split % argument.


class CamelsGB(Dataset):
    """
    Create a PyTorch `Dataset` containing data of basin(s) from CAMELS-GB.

    CAMELS-GB contains forcing data (precipitation, temperature etc.) and
    discharge data for 671 hydrological basins/catchments in the UK. This class
    loads data from an arbitrary number of these basins (by default all 671).
    """
    def __init__(self, seq_length: int = 365, basin_ids: Optional[List[str]] = None, mode: Optional[str] = None,
                 dates: Optional[List[pd.Timestamp]] = None, means: Optional[Dict[str, float]] = None,
                 stds: Optional[Dict[str, float]] = None) -> None:
        """
        Initialise dataset containing the data of basin(s) from CAMELS-GB.

        By default, this class loads the data from all 671 basins in the
        dataset. Alternatively, a list of string basin IDs can be passed to
        the `basin_ids` argument to selectively load data from a small number of
        basins.

        Args:
            seq_length (int, optional): Length of the time window of
            meteorological input provided for one time step of prediction.
            Defaults to 365.
            basin_ids (str, optional): List of string IDs of the basins to load
            data from. If `None`, will load all 671 basins. Defaults to `None`.
            mode (str, optional): One of `['train', 'eval', None]`. If `None`,
            the entire time series will be loaded. Defaults to `None`.
            dates (List, optional):  List of `pd.DateTime` dates of the start
            and end of the discharge mode. Defaults to `None`.
            means: (Dict, optional) Means of input and output features derived
            from the training period. Has to be provided for `mode='eval'`. Can
            be retrieved by calling `get_means()` on the dataset.
            stds: (Dict, optional) Std of input and output features derived
            from the training period. Has to be provided for `mode='eval'`. Can
            be retrieved by calling `get_stds()` on the dataset.
        """
        self.seq_length: int = seq_length
        self.basin_ids: Optional[List[str]] = basin_ids
        self.mode: Optional[str] = mode
        self.dates: Optional[List[pd.Timestamp]] = dates

        if mode != 'train' and means is not None and stds is not None:
            self.means: Dict[str, float] = means
            self.stds: Dict[str, float] = stds
        elif mode != 'train' and means is None or stds is None:
            raise TypeError("When mode is not 'train' it is necessary to pass values to both `means` and `stds`.")

        self.x, self.y = self._load_data()

        self.num_samples: int = self.x.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def get_means(self) -> Dict[str, float]:
        return self.means

    def get_stds(self) -> Dict[str, float]:
        return self.stds

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Load input and output data from the forcing and discharge datasets
        directory = os.path.join(os.getcwd(), r'src\data\CAMELS-GB\timeseries')
        filename = 'CAMELS_GB_hydromet_timeseries_1001_19701001-20150930.csv'
        df: pd.DataFrame = pd.read_csv(os.path.join(directory, filename))
        data: pd.DataFrame = df[['precipitation', 'pet', 'temperature', 'peti', 'humidity', 'shortwave_rad',
                                  'longwave_rad', 'windspeed', 'date']]
        data.date = pd.to_datetime(data.date, dayfirst=True, format="%Y/%m/%d")
        data['QObs(mm/d)'] = df['discharge_spec']
        del df
        # TODO: store fully processed file of all 671 basins combined that will be saved the first time this is run?

        if self.dates is not None:
            # If meteorological observations exist before start date
            # use these as well. Similiar to hydrological warmup mode.
            if self.dates[0] - pd.DateOffset(days=self.seq_length) > data.index[0]:
                start_date: pd.Timestamp = self.dates[0] - pd.DateOffset(days=self.seq_length)
            else:
                start_date = self.dates[0]
            data = data[start_date:self.dates[1]]

        # if training mode store means and stds
        if self.mode == 'train':
            self.means = data.mean().to_dict()
            self.stds = data.std().to_dict()

        # extract input and output features from data dataframe loaded above
        x: np.ndarray = data[['precipitation', 'shortwave_rad', 'temperature', 'peti', 'humidity']].to_numpy()
        y: np.ndarray = data['QObs(mm/d)'].to_numpy()

        # normalise data, reshape for LSTM training and remove invalid samples
        # normalisation is done because it speeds up training and allows the model to give appropriate weight to each
        # feature
        x = self._local_normalization(x, variable='inputs')
        x, y = self._reshape_data(x, y)

        if self.mode == "train":
            # data tidying!
            # Delete all samples where discharge is NaN
            if np.sum(np.isnan(y)) > 0:
                print("Deleted some records because of NaNs")
                x = np.delete(x, np.argwhere(np.isnan(y)), axis=0)
                y = np.delete(y, np.argwhere(np.isnan(y)), axis=0)

            # Deletes all records where no discharge was measured (-999)
            x = np.delete(x, np.argwhere(y < 0)[:, 0], axis=0)
            y = np.delete(y, np.argwhere(y < 0)[:, 0], axis=0)

            # normalise discharge - same reasons as given above for the inputs
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))

    def _fetch_basins(self, basin_ids: List[str]) -> pd.DataFrame:
        raise NotImplementedError

    def _local_normalization(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """
        Normalize input/output features with local mean/std.

        Args:
            feature (np.ndarray): Array containing the feature(s) as a matrix.
            variable (str): Either `inputs` or `output` showing which feature
            will be normalized.

        Raises:
            RuntimeError: If `variable` is not `inputs` or `output`.

        Returns:
            np.ndarray: Array of the same shape as `feature` containing the
            normalized features.
        """
        if variable == 'inputs':
            means = np.array([self.means['precipitation'],
                              self.means['shortwave_rad'],
                              self.means['temperature'],
                              self.means['peti'],
                              self.means['humidity']])
            stds = np.array([self.stds['precipitation'],
                             self.stds['shortwave_rad'],
                             self.stds['temperature'],
                             self.stds['peti'],
                             self.stds['humidity']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = (feature - self.means["QObs(mm/d)"]) / self.stds["QObs(mm/d)"]
        else:
            raise RuntimeError(f"Unknown variable type {type(variable)}")

        return feature

    def local_rescale(self, feature: np.ndarray, variable: str) -> np.ndarray:
        """
        Rescale input/output features back to original size with local mean/std.

        Args:
            feature (np.ndarray): Array containing the feature(s) as a matrix.
            variable (str): Either `inputs` or `output` showing which feature
            will be normalized.

        Raises:
            RuntimeError: If `variable` is not `inputs` or `output`.

        Returns:
            np.ndarray: Array of the same shape as `feature` containing the
            normalized features.
        """
        if variable == 'inputs':
            means = np.array([self.means['precipitation'],
                              self.means['shortwave_rad'],
                              self.means['temperature'],
                              self.means['peti'],
                              self.means['humidity']])
            stds = np.array([self.stds['precipitation'],
                             self.stds['shortwave_rad'],
                             self.stds['temperature'],
                             self.stds['peti'],
                             self.stds['humidity']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = feature * self.stds["QObs(mm/d)"] + self.means["QObs(mm/d)"]
        else:
            raise RuntimeError(f"Unknown variable type {type(variable)}")

        return feature

    def _reshape_data(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape matrix data into sample shape for LSTM training.

        Args:
            x (np.ndarray): Matrix containing input features column wise and
            time steps row wise.
            y (np.ndarray): Matrix containing the output feature.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two np.ndarrays, the first of shape
            `(num_samples, seq_length, num_features)`, containing the
            input data for the LSTM, the second of shape `(num_samples, 1)`
            containing the expected output for each input sample.
        """
        num_samples, num_features = x.shape

        x_new = np.zeros((num_samples - self.seq_length + 1, self.seq_length, num_features))
        y_new = np.zeros((num_samples - self.seq_length + 1, 1))

        for i in range(0, x_new.shape[0]):
            x_new[i, :, :num_features] = x[i:i + self.seq_length, :]
            y_new[i, :] = y[i + self.seq_length - 1]

        return x_new, y_new
