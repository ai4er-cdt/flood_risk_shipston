# Place all your constants here
import os
import pickle
from typing import Tuple

# Note: constants should be UPPER_CASE
constants_path = os.path.realpath(__file__)
SRC_PATH = os.path.dirname(constants_path)
PROJECT_PATH = os.path.dirname(SRC_PATH)
SAVE_PATH = os.path.join(PROJECT_PATH, 'logs')
os.makedirs(SAVE_PATH, exist_ok=True)

# Load sorted tuple of all basin ids
with open(os.path.join(SRC_PATH, 'constants.pickle'), 'rb') as file:
    ALL_BASINS: Tuple[int] = pickle.load(file)
