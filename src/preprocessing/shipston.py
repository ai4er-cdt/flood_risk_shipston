import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from numba import njit
from src.constants import *
from src.preprocessing import BaseDataset


class ShipstonDataset(BaseDataset):
    """Create a PyTorch `Dataset` containing timeseries data from Shipston."""
    def __init__(self, features: Dict[str, List[str]], dates: List[str], test_end_date: str, train: bool = True,
                 seq_length: int = 365, train_test_split: str = '2010') -> None:
        """
        Initialise dataset containing the timeseries data from Shipston.

        Args:
            features (Dict): Dictionary where the keys are the feature type and
            the values are lists of strings with the features to include in the
            dataset.
            dates (List): List of specified string dates for the start and end
            of the discharge. This overrides the `train_test_split` and `train`
            parameters.
            test_end_date (str): Specifies the end date of the test dataset.
            train (bool, optional): If `True`, creates dataset from the training
            set, otherwise creates from the test set. Defaults to `True`.
            seq_length (int, optional): Length of the time window of
            meteorological input provided for one time step of prediction.
            train_test_split (str, optional): Date to split the data into the
            train and test sets. Discharge values from before this date will be
            part of the training set, those after will be the test set. Specific
            days should be passed in the format `YYYY-MM-DD`, years can be
            passed as `YYYY`. Defaults to `'2010-01-01'`.
        """
        self.features: List[str] = features['timeseries']
        self.dates: List = [pd.Timestamp(date) for date in dates]
        self.test_end_date: str = test_end_date
        self.train: bool = train
        self.seq_length: int = seq_length
        self.train_test_split: pd.Timestamp = pd.Timestamp(train_test_split)

        # Suppress Numba deprecation warning.
        warnings.filterwarnings("ignore", message="\nEncountered the use of a type that is scheduled for ")

        self.x, self.y = self._load_data()

        self.num_samples: int = self.x.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        timeseries_columns: List[str] = ['date'] + list(self.features) + ['discharge_vol']
        data = pd.read_csv(os.path.join(DATA_PATH, SHIPSTON_ID), usecols=timeseries_columns,
                           parse_dates=[0], infer_datetime_format=True, dtype=np.float32)
        # Crop the date range as much as possible.
        if len(self.dates) == 0 and self.train:
            self.dates = [data.date[0], self.train_test_split]
        elif len(self.dates) == 0 and not self.train:
            self.dates = [self.train_test_split, self.test_end_date]
        data = self._crop_dates(data, start_date=self.dates[0], end_date=self.dates[1])
        # Remove as many contiguous regions of NaNs as possible.
        data = self._remove_nan_regions(data)

        # List of feature names in `data` with a constant ordering independent of `data` or the features dict.
        self.feature_names: List[str] = [col for col in SHIPSTON_FEATURES if col in list(data.columns)]

        # Extract input and output features from dataframe loaded above.
        x: np.ndarray = data[self.feature_names].to_numpy(dtype=np.float32)
        y: np.ndarray = data['discharge_vol'].to_numpy(dtype=np.float32)

        # Normalise data, reshape for LSTM training and remove invalid samples.
        x = self._normalization(x, input=True)
        x, y = _reshape_data(x, y, self.seq_length, np.float32)
        if self.train:
            # Normalise discharge - only needs to be done when training.
            y = self._normalization(y, input=False)
        # Convert arrays to torch tensors.
        return torch.from_numpy(x), torch.from_numpy(y)

    def _normalization(self, data: np.ndarray, input: bool) -> np.ndarray:
        """
        Normalize input/output features with mean/std from across all basins.

        Args:
            data (np.ndarray): Array containing the features as a matrix.
            input (bool): If True, the `data` array is the model input.

        Returns:
            np.ndarray: Array of the same shape as `data` containing the
            normalized features.
        """
        if input:
            means = np.array([SHIPSTON_STATISTICS[feature][0] for feature in self.feature_names])
            stds = np.array([SHIPSTON_STATISTICS[feature][1] for feature in self.feature_names])
            data = (data - means) / stds
        else:
            data = ((data - SHIPSTON_STATISTICS["discharge_vol"][0]) / SHIPSTON_STATISTICS["discharge_vol"][1])

        return data

    def rescale(self, data: torch.Tensor, input: bool) -> torch.Tensor:
        """
        Rescale input/output features back to original size with mean/std.

        The mean/std is again calculated across all 671 basins.

        Args:
            data (torch.Tensor): Array containing the features as a matrix.
            input (bool): If True, the `data` array is the model input.

        Returns:
            torch.Tensor: Array of the same shape as `data` containing the
            normalized features.
        """
        if input:
            means = torch.tensor([SHIPSTON_STATISTICS[feature][0] for feature in self.feature_names],
                                 dtype=torch.float32)
            stds = torch.tensor([SHIPSTON_STATISTICS[feature][1] for feature in self.feature_names],
                                dtype=torch.float32)
            data = data * stds + means
        else:
            data = (data * SHIPSTON_STATISTICS["discharge_vol"][1] + SHIPSTON_STATISTICS["discharge_vol"][0])

        return data

    def _crop_dates(self, df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Remove dates in df outside the range from `start_date` to `end_date`.

        Assumes data in `df` is in chronological order. Also keeps a leeway of
        `seq_length` days before `start_date` so that we can start training from
        that date. The end date itself is not included.

        Args:
            df (pd.DataFrame): Input data with dates in chronological order.
            start_date (pd.Timestamp): Start date of discharge period.
            end_date (pd.Timestamp): End date of discharge period.

        Returns:
            pd.DataFrame: The input dataframe with dates cropped to a range.
        """
        if start_date - pd.DateOffset(days=self.seq_length) > df.date[0]:
            start_date -= pd.DateOffset(days=self.seq_length)

        start_index = df.loc[df['date'] == start_date].index[0]
        end_index = df.loc[df['date'] == end_date].index[0]
        return df[start_index:end_index]

    def _remove_nan_regions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove some regions of `df` where the discharge data contains NaNs.

        This function assumes the data comes from a single basin and is in
        chronological order. Note that because we don't want to remove the
        previous `seq_length` days of forcing data for the first discharge value
        after a sequence of NaNs, we can't remove all rows with NaNs. Therefore
        we only remove the rows with NaNs which have more than `seq_length`
        consecutive NaNs in front.

        Args:
            df (pd.DataFrame): Input dataframe.

        Returns:
            pd.DataFrame: The input dataframe with some rows of NaNs removed.
        """
        nan_regions = []
        in_nanregion = False
        # Calculate the start and end indices of all sections of nans in the discharge data.
        for row in range(df.index[0], df.index[-1] + 1):
            if pd.isna(df['discharge_vol'][row]) and not in_nanregion:
                nan_regions.append(row)
                in_nanregion = True
            if not pd.isna(df['discharge_vol'][row]) and in_nanregion:
                nan_regions.append(row - 1)
                in_nanregion = False
        # Remove the rows with nans.
        for idx in range(1, len(nan_regions), 2):
            # There are three hard things in programming - naming things and off-by-one errors. :-)
            start_nan = nan_regions[idx - 1]
            end_nan = nan_regions[idx] - self.seq_length + 1
            df = df.drop(df.index[start_nan:end_nan + 1])
        return df


@njit
def _reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    The Numba decorator compiles this function to machine code, speeding up the
    execution time by several orders of magnitude (cuts loading from 10 minutes
    to 10 seconds using half of the basins). This function needs to be outside
    of the class because Numba does not recognise the class type.

    Args:
        x (np.ndarray): Matrix containing input features column wise and time
        steps row wise.
        y (np.ndarray): Matrix containing the output feature.
        seq_length (int): Length of the time window of meteorological input
        provided for one time step of prediction.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two np.ndarrays, the first of shape
        `(num_samples, seq_length, num_features)`, containing the
        input data for the LSTM, the second of shape `(num_samples, 1)`
        containing the expected output for each input sample.
    """
    _, num_features = x.shape
    # Iterate once through all time steps to calculate number of valid data points.
    # This is necessary because of short sections of NaNs in the discharge data.
    num_samples = 0
    for i in range(seq_length - 1, len(x)):
        if not np.isnan(y[i]):
            num_samples += 1
    # Assign empty numpy arrays with the correct size.
    x_new = np.empty((num_samples, seq_length, num_features), dtype=np.float32)
    y_new = np.empty((num_samples, 1), dtype=np.float32)

    num_samples = 0  # Start new counter so we can index the new arrays.
    for i in range(seq_length - 1, len(x)):
        if not np.isnan(y[i]):
            x_new[num_samples, :, :num_features] = x[i - seq_length + 1:i + 1, :]
            y_new[num_samples, :] = y[i]
            num_samples += 1

    return x_new, y_new
