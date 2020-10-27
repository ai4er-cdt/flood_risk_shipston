# Place all your constants here
import json
import os
from typing import Dict, List, Tuple

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
SAVE_PATH = os.path.join(PROJECT_PATH, 'logs')
os.makedirs(SAVE_PATH, exist_ok=True)
# Dropbox URL of dataset.
DATASET_URL = "https://www.dropbox.com/s/4x7db07gjakl7sk/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip?raw=1"
# Filename of dataset
DATASET_ID = "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
# Load JSON with dataset information.
with open(os.path.join(SRC_PATH, 'constants.json')) as json_file:
    data_dict = json.load(json_file)

# Load sorted list of all basin ids
ALL_BASINS: List[int] = data_dict['ALL_BASINS']
# List of all feature names except the non-timeseries features containing NaNs.
# This is to make sure we are not using basin attribute features with NaNs.
ALL_FEATURES: List[str] = data_dict['ALL_FEATURES']
# Dict containing the means and stds of every feature in CAMELS-GB, calculated
# across all 671 basins.
FEATURE_STATISTICS: Dict[str, List[float]] = data_dict['FEATURE_STATISTICS']
# Tuple containing the different sources of data in CAMELS-GB.
DATASET_KEYS: Tuple = ('timeseries', 'climatic', 'humaninfluence', 'hydrogeology',
                       'hydrologic', 'hydrometry', 'landcover', 'soil', 'topographic')
