import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from src import constants
from torch.utils.data import Dataset


class CamelsGB(Dataset):
    """
    Create a PyTorch `Dataset` containing data of basin(s) from CAMELS-GB.

    CAMELS-GB contains forcing data (precipitation, temperature etc.) and
    discharge data for 671 hydrological basins/catchments in the UK. This class
    loads data from an arbitrary number of these basins (by default all 671).
    """
    def __init__(self, data_dir: str, features: Dict[str, List[str]], basins_frac: float, dates: List[str],
                 train: bool = True, seq_length: int = 365, train_test_split: str = '2010',
                 means: Optional[Dict[str, float]] = None, stds: Optional[Dict[str, float]] = None) -> None:
        """
        Initialise dataset containing the data of basin(s) from CAMELS-GB.

        By default, this class loads the data from all 671 basins in the
        dataset. Alternatively, a list of string basin IDs can be passed to
        the `basin_ids` argument to selectively load data from a subset of the
        basins.

        Args:
            data_dir (str): Path to the directory with the data.
            features (Dict): Dictionary where the keys are the feature type and
            the values are lists of strings with the features to include in the
            dataset.
            basins_frac (float): Fraction of basins to load data from. 1.0 will
            load all 671 basins, 0.0 will load none.
            dates (List):  List of string dates of the start and end of the
            discharge mode. This overrides the `train_test_split` and `train`
            parameters.
            train (bool, optional): If `True`, creates dataset from the training
            set, otherwise creates from the test set. Defaults to `True`.
            seq_length (int): Length of the time window of meteorological input
            provided for one time step of prediction.
            train_test_split (str): Date to split the data into the
            train and test sets. Discharge values from before this date will be
            part of the training set, those after will be the test set. Specific
            days should be passed in the format `YYYY-MM-DD`, years can be
            passed as `YYYY`. Defaults to `'2010-01-01'`.
            means: (Dict, optional): Means of input and output features derived
            from the training period. Has to be provided when `train=False`. Can
            be retrieved by calling `get_means()` on the dataset.
            stds: (Dict, optional): Std of input and output features derived
            from the training period. Has to be provided when `train=False`. Can
            be retrieved by calling `get_stds()` on the dataset.
        """
        self.data_dir: str = data_dir
        # Use defaultdict to avoid errors when we ask for a key that isn't in the dict.
        self.features: Dict[str, List[str]] = defaultdict(list, features)
        self.train: bool = train
        self.seq_length: int = seq_length
        self.train_test_split: pd.Timestamp = pd.Timestamp(train_test_split)
        self.dates: List = [pd.Timestamp(date) for date in dates]
        self.basin_ids: List[int] = list(constants.ALL_BASINS[:int(len(constants.ALL_BASINS) * basins_frac)])
        # Remove two particular basins from the list if we use either of these features since these are the only two
        # basins with NaN values for these potentially useful features.
        if 'dpsbar' in self.features['topographic'] or 'elem_mean' in self.features['topographic']:
            for basin_id in (18011, 26006):
                if basin_id in self.basin_ids:
                    self.basin_ids.remove(basin_id)

        if not self.train and means is not None and stds is not None:
            self.means: Dict[str, float] = means
            self.stds: Dict[str, float] = stds
        elif not self.train and (means is None or stds is None):
            raise TypeError("When `train=False` it is necessary to pass values to both `means` and `stds`.")

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
        first_basin: int = self.basin_ids[0]
        filename: str = f'CAMELS_GB_hydromet_timeseries_{first_basin}_19701001-20150930.csv'
        timeseries_columns: List[str] = ['date'] + list(self.features['timeseries']) + ['discharge_spec']

        data: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, 'timeseries', filename),
                                         usecols=timeseries_columns, parse_dates=[0], infer_datetime_format=True)
        data.rename(columns={"discharge_spec": "QObs(mm/d)"}, inplace=True)
        data.date = pd.to_datetime(data.date, dayfirst=True, format="%Y-%m-%d")
        data['basin_id'] = first_basin

        for key in constants.DATASET_KEYS[1:]:
            filename = f'CAMELS_GB_{key}_attributes.csv'
            attr_df: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, filename),
                                                usecols=['gauge_id'] + list(self.features[key]),
                                                index_col='gauge_id')
            for name, row in attr_df.loc[first_basin][self.features[key]].iteritems():
                data[name] = row

        if len(self.dates) == 0 and self.train:
            self.dates = [data.date[0], self.train_test_split]
        elif len(self.dates) == 0 and not self.train:
            self.dates = [self.train_test_split, data.date.iloc[-1]]
        data = self._crop_dates(data, start_date=self.dates[0], end_date=self.dates[1])
        data = self._remove_nan_regions(data)
        # List to keep track of start and end indexes of each basin in the final array.
        basin_indexes: List[Tuple] = [(0, len(data))]

        if len(self.basin_ids) > 1:
            for basin_idx in range(1, len(self.basin_ids)):
                basin: int = self.basin_ids[basin_idx]
                filename = f'CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv'

                new_data: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, 'timeseries', filename),
                                                     usecols=timeseries_columns, parse_dates=[0],
                                                     infer_datetime_format=True)
                new_data.rename(columns={"discharge_spec": "QObs(mm/d)"}, inplace=True)
                new_data.date = pd.to_datetime(new_data.date, dayfirst=True, format="%Y-%m-%d")
                new_data['basin_id'] = basin

                for key in constants.DATASET_KEYS[1:]:
                    filename = f'CAMELS_GB_{key}_attributes.csv'
                    attr_df = pd.read_csv(os.path.join(self.data_dir, filename),
                                          usecols=['gauge_id'] + list(self.features[key]),
                                          index_col='gauge_id')
                    for name, row in attr_df.loc[basin][self.features[key]].iteritems():
                        data[name] = row

                if len(self.dates) == 0 and self.train:
                    self.dates = [new_data.date[0], self.train_test_split]
                elif len(self.dates) == 0 and not self.train:
                    self.dates = [self.train_test_split, new_data.date.iloc[-1]]
                new_data = self._crop_dates(new_data, start_date=self.dates[0], end_date=self.dates[1])
                new_data = self._remove_nan_regions(new_data)

                basin_indexes.append((basin_indexes[-1][1], basin_indexes[-1][1] + len(new_data)))
                data = pd.concat([data, new_data], axis=0, ignore_index=True)
                del new_data

        # Feature names in `data` with a constant ordering independent of `data` or the features dict.
        self.feature_names: List[str] = [col for col in constants.ALL_FEATURES if col in list(data.columns)]
        # TODO: Consider storing means and stds from all basins combined and only using those.
        # If training mode store means and stds.
        if self.train:
            self.means = data[self.feature_names + ['QObs(mm/d)']].mean().to_dict()
            self.stds = data[self.feature_names + ['QObs(mm/d)']].std().to_dict()

        # Extract input and output features from dataframe loaded above.
        x: np.ndarray = data[self.feature_names].to_numpy(dtype=np.float32)
        y: np.ndarray = data['QObs(mm/d)'].to_numpy(dtype=np.float32)

        # Normalise data, reshape for LSTM training and remove invalid samples.
        x = self._local_normalization(x, variable='inputs')
        x, y = self._reshape_data(x, y, basin_indexes)

        if self.train:
            # Normalise discharge - only needs to be done when training.
            y = self._local_normalization(y, variable='output')

        # Convert arrays to torch tensors.
        return torch.from_numpy(x), torch.from_numpy(y)

    def _local_normalization(self, data_array: np.ndarray, variable: str) -> np.ndarray:
        """
        Normalize input/output features with local mean/std.

        Args:
            data_array (np.ndarray): Array containing the features as a matrix.
            variable (str): Either `'inputs'` or `'output'` depending on which
            feature will be normalized.

        Raises:
            RuntimeError: If `variable` is not `'inputs'` or `'output'`.

        Returns:
            np.ndarray: Array of the same shape as `data_array` containing the
            normalized features.
        """
        if variable == 'inputs':
            means = np.array([self.means[feature] for feature in self.feature_names])
            stds = np.array([self.stds[feature] for feature in self.feature_names])
            data_array = (data_array - means) / stds
        elif variable == 'output':
            data_array = (data_array - self.means["QObs(mm/d)"]) / self.stds["QObs(mm/d)"]
        else:
            raise TypeError(f"Unknown variable type {type(variable)}")

        return data_array

    def local_rescale(self, data_array: torch.Tensor, variable: str) -> torch.Tensor:
        """
        Rescale input/output features back to original size with local mean/std.

        Args:
            data_array (torch.Tensor): Array containing the features as a matrix.
            variable (str): Either `'inputs'` or `'output'` depending on which
            feature will be normalized.

        Raises:
            RuntimeError: If `variable` is not `'inputs'` or `'output'`.

        Returns:
            torch.Tensor: Array of the same shape as `data_array` containing the
            normalized features.
        """
        # TODO: Check shapes.
        if variable == 'inputs':
            means = torch.tensor([self.means[feature] for feature in self.feature_names], dtype=torch.float32)
            stds = torch.tensor([self.stds[feature] for feature in self.feature_names], dtype=torch.float32)
            data_array = data_array * stds + means
        elif variable == 'output':
            data_array = data_array * self.stds["QObs(mm/d)"] + self.means["QObs(mm/d)"]
        else:
            raise TypeError(f"Unknown variable type {type(variable)}")

        return data_array

    def _reshape_data(self, x: np.ndarray, y: np.ndarray, basin_indexes: List[Tuple]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape matrix data into sample shape for LSTM training.

        Args:
            x (np.ndarray): Matrix containing input features column wise and
            time steps row wise.
            y (np.ndarray): Matrix containing the output feature.
            basin_indexes (List): List of tuples containing the start and end
            indexes for each basin, in the form (start_idx, end_idx).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two np.ndarrays, the first of shape
            `(num_samples, seq_length, num_features)`, containing the
            input data for the LSTM, the second of shape `(num_samples, 1)`
            containing the expected output for each input sample.
        """
        _, num_features = x.shape
        # Iterate once through all time steps for each basin to calculate number of valid data points.
        # This is necessary because of short sections of NaNs in the discharge data.
        num_samples = 0
        for (start_idx, end_idx) in basin_indexes:
            for i in range(start_idx + self.seq_length - 1, end_idx):
                if not np.isnan(y[i]):
                    num_samples += 1
        # Assign empty numpy arrays with the correct size.
        x_new = np.empty((num_samples, self.seq_length, num_features), dtype=np.float32)
        y_new = np.empty((num_samples, 1), dtype=np.float32)

        num_samples = 0  # Start new counter so we can index the new arrays.
        for (start_idx, end_idx) in basin_indexes:
            for i in range(start_idx + self.seq_length - 1, end_idx):
                if not np.isnan(y[i]):
                    x_new[num_samples, :, :num_features] = x[i - self.seq_length + 1:i + 1, :]
                    y_new[num_samples, :] = y[i]
                    num_samples += 1

        return x_new, y_new

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
            if pd.isna(df['QObs(mm/d)'][row]) and not in_nanregion:
                nan_regions.append(row)
                in_nanregion = True
            if not pd.isna(df['QObs(mm/d)'][row]) and in_nanregion:
                nan_regions.append(row - 1)
                in_nanregion = False
        # Remove the rows with nans.
        for idx in range(1, len(nan_regions), 2):
            # There are three hard things in programming - naming things and off-by-one errors. :-)
            start_nan = nan_regions[idx - 1]
            end_nan = nan_regions[idx] - self.seq_length + 1
            df.drop(df.index[start_nan:end_nan + 1], inplace=True)
        return df
