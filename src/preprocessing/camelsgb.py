import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from numba import njit
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
                 precision: int = 32) -> None:
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
            seq_length (int, optional): Length of the time window of
            meteorological input provided for one time step of prediction.
            train_test_split (str, optional): Date to split the data into the
            train and test sets. Discharge values from before this date will be
            part of the training set, those after will be the test set. Specific
            days should be passed in the format `YYYY-MM-DD`, years can be
            passed as `YYYY`. Defaults to `'2010-01-01'`.
            precision (int, optional): Whether to load data as single (32-bit)
            or half (16-bit) precision floating points. Can only be 16 or 32.
            Defaults to 32.
        """
        self.data_dir: str = os.path.join(data_dir, 'CAMELS-GB')
        # Use defaultdict to avoid errors when we ask for a key that isn't in the dict.
        self.features: Dict[str, List[str]] = defaultdict(list, features)
        self.train: bool = train
        self.seq_length: int = seq_length
        self.train_test_split: pd.Timestamp = pd.Timestamp(train_test_split)
        self.dates: List = [pd.Timestamp(date) for date in dates]
        self.precision = np.float32 if precision == 32 else np.float16
        if self.train:
            self.basin_ids: List[int] = list(constants.ALL_BASINS[:round(len(constants.ALL_BASINS) * basins_frac)])
            # Remove two particular basins from the list if we use either of these features since these are the only two
            # basins with NaN values for these potentially useful features.
            if 'dpsbar' in self.features['topographic'] or 'elem_mean' in self.features['topographic']:
                for basin_id in (18011, 26006):
                    if basin_id in self.basin_ids:
                        self.basin_ids.remove(basin_id)
        else:
            self.basin_ids = [constants.ALL_BASINS[2]]

        self.x, self.y = self._load_data()

        self.num_samples: int = self.x.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]

    def _load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        timeseries_columns: List[str] = ['date'] + list(self.features['timeseries']) + ['discharge_spec']
        basin_indexes: List[Tuple] = []
        df_list: List[pd.DataFrame] = []

        def process_df(df: pd.DataFrame, basin_indexes: List[Tuple], loop_idx: int) -> pd.DataFrame:
            df.rename(columns={"discharge_spec": "QObs(mm/d)"}, inplace=True)
            df.date = pd.to_datetime(df.date, dayfirst=True, format="%Y-%m-%d")
            df['basin_id'] = basin

            for key in constants.DATASET_KEYS[1:]:
                if len(self.features[key]) > 0:
                    filename = f'CAMELS_GB_{key}_attributes.csv'
                    attr_df: pd.DataFrame = pd.read_csv(os.path.join(self.data_dir, filename),
                                                        usecols=['gauge_id'] + list(self.features[key]),
                                                        index_col='gauge_id')
                    for name, row in attr_df.loc[basin][self.features[key]].iteritems():
                        if name == 'dom_land_cover':
                            # Label encoding is needed for only this attribute (in the landcover data).
                            dom_land_cover_dict = {"Grass and Pasture": 0, "Shrubs": 1, "Crops": 2, "Urban": 3,
                                                "Deciduous Woodland": 4, "Evergreen Woodland": 5}
                            row = dom_land_cover_dict[row]
                        df[name] = row
            if len(self.dates) == 0 and self.train:
                self.dates = [df.date[0], self.train_test_split]
            elif len(self.dates) == 0 and not self.train:
                self.dates = [self.train_test_split, df.date.iloc[-1]]
            df = self._crop_dates(df, start_date=self.dates[0], end_date=self.dates[1])
            df = self._remove_nan_regions(df)
            if loop_idx == 0:
                basin_indexes.append((0, len(df)))
            else:
                basin_indexes.append((basin_indexes[-1][1], basin_indexes[-1][1] + len(df)))
            return df

        for basin_idx in range(len(self.basin_ids)):
            basin: int = self.basin_ids[basin_idx]
            filepath: str = os.path.join(self.data_dir, 'timeseries',
                                         f'CAMELS_GB_hydromet_timeseries_{basin}_19701001-20150930.csv')
            df_list.append(process_df(pd.read_csv(filepath, usecols=timeseries_columns, parse_dates=[0],
                                                    infer_datetime_format=True), basin_indexes, basin_idx))

        data = pd.concat(df_list, axis=0, ignore_index=True)

        # List of feature names in `data` with a constant ordering independent of `data` or the features dict.
        self.feature_names: List[str] = [col for col in constants.ALL_FEATURES if col in list(data.columns)]

        # Extract input and output features from dataframe loaded above.
        x: np.ndarray = data[self.feature_names].to_numpy(dtype=self.precision)
        y: np.ndarray = data['QObs(mm/d)'].to_numpy(dtype=self.precision)

        # Normalise data, reshape for LSTM training and remove invalid samples.
        x = self._local_normalization(x, variable='inputs')
        x, y = _reshape_data(x, y, basin_indexes, self.seq_length, self.precision)

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
            means = np.array([constants.FEATURE_STATISTICS[feature][0] for feature in self.feature_names])
            stds = np.array([constants.FEATURE_STATISTICS[feature][1] for feature in self.feature_names])
            data_array = (data_array - means) / stds
        elif variable == 'output':
            data_array = ((data_array - constants.FEATURE_STATISTICS["QObs(mm/d)"][0]) /
                          constants.FEATURE_STATISTICS["QObs(mm/d)"][1])
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
        if variable == 'inputs':
            means = torch.tensor([constants.FEATURE_STATISTICS[feature][0] for feature in self.feature_names],
                                 dtype=torch.float32)
            stds = torch.tensor([constants.FEATURE_STATISTICS[feature][1] for feature in self.feature_names],
                                dtype=torch.float32)
            data_array = data_array * stds + means
        elif variable == 'output':
            data_array = (data_array * constants.FEATURE_STATISTICS["QObs(mm/d)"][1] +
                          constants.FEATURE_STATISTICS["QObs(mm/d)"][0])
        else:
            raise TypeError(f"Unknown variable type {type(variable)}")

        return data_array

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
            df = df.drop(df.index[start_nan:end_nan + 1])
        return df


@njit
def _reshape_data(x: np.ndarray, y: np.ndarray, basin_indexes: List[Tuple], seq_length: int,
                  precision: Any) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    The numba decorator compiles this function to machine code, achieving
    potentially large speedups. This function needs to be outside of the class
    because Numba does not recognise the class type.

    Args:
        x (np.ndarray): Matrix containing input features column wise and
        time steps row wise.
        y (np.ndarray): Matrix containing the output feature.
        basin_indexes (List): List of tuples containing the start and end
        indexes for each basin, in the form (start_idx, end_idx).
        seq_length (int): Length of the time window of
        meteorological input provided for one time step of prediction.
        precision (int): Whether to load data as `np.float32` or `np.float16`.

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
        for i in range(start_idx + seq_length - 1, end_idx):
            if not np.isnan(y[i]):
                num_samples += 1
    # Assign empty numpy arrays with the correct size.
    x_new = np.empty((num_samples, seq_length, num_features), dtype=precision)
    y_new = np.empty((num_samples, 1), dtype=precision)

    num_samples = 0  # Start new counter so we can index the new arrays.
    for (start_idx, end_idx) in basin_indexes:
        for i in range(start_idx + seq_length - 1, end_idx):
            if not np.isnan(y[i]):
                x_new[num_samples, :, :num_features] = x[i - seq_length + 1:i + 1, :]
                y_new[num_samples, :] = y[i]
                num_samples += 1

    return x_new, y_new
