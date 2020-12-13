import json
import os
from typing import Dict, List, Tuple

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
DATA_PATH = os.path.join(SRC_PATH, "data")
SAVE_PATH = os.path.join(PROJECT_PATH, 'logs')
os.makedirs(SAVE_PATH, exist_ok=True)
# Dropbox URL of CAMELS-GB dataset.
CAMELS_URL = "https://www.dropbox.com/s/4x7db07gjakl7sk/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip?raw=1"
# Filename of CAMELS-GB dataset.
CAMELS_ID = "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
# Dropbox URL of Shipston time series dataset.
SHIPSTON_URL = "https://www.dropbox.com/s/hv0cnv3q3i8rbpk/shipstonv4.csv?raw=1"
SHIPSTON_ID = SHIPSTON_URL[SHIPSTON_URL.index('shipston'):SHIPSTON_URL.index('csv') + 3]
# Load JSON with dataset information.
with open(os.path.join(SRC_PATH, 'constants.json')) as json_file:
    data_dict = json.load(json_file)

# Load randomised list of all basin ids
ALL_BASINS: List[int] = data_dict['ALL_BASINS']
# List of all feature names except the non-timeseries features containing NaNs.
# This is to make sure we are not using basin attribute features with NaNs.
CAMELS_FEATURES: List[str] = data_dict['CAMELS_FEATURES']
SHIPSTON_FEATURES: List[str] = data_dict['SHIPSTON_FEATURES']
# Dict containing the means and stds of every feature in CAMELS-GB, calculated across all 671 basins.
FEATURE_STATISTICS: Dict[str, List[float]] = data_dict['FEATURE_STATISTICS']
# Dict containing means and stds of timeseries features from Shipston data.
SHIPSTON_STATISTICS: Dict[str, List[float]] = data_dict['SHIPSTON_STATISTICS']
# Tuple containing the different sources of data in CAMELS-GB.
DATASET_KEYS: Tuple = ('timeseries', 'climatic', 'humaninfluence', 'hydrogeology',
                       'hydrologic', 'hydrometry', 'landcover', 'soil', 'topographic')
