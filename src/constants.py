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
# Dropbox URL of dataset.
DATASET_URL = "https://www.dropbox.com/s/4x7db07gjakl7sk/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip?raw=1"
# Filename of dataset
DATASET_ID = "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
# Load sorted tuple of all basin ids
with open(os.path.join(SRC_PATH, 'constants.pickle'), 'rb') as file:
    ALL_BASINS: Tuple[int] = pickle.load(file)
# Tuple containing the different sources of data in CAMELS-GB.
DATASET_KEYS = ('timeseries', 'climatic', 'humaninfluence', 'hydrogeology',
                'hydrologic', 'hydrometry', 'landcover', 'soil', 'topographic')
# Tuple of all feature names except the non-timeseries features containing NaNs.
# This is to make sure we are not using basin attribute features with NaNs.
ALL_FEATURES = ('precipitation', 'pet', 'temperature', 'peti', 'humidity', 'shortwave_rad', 'longwave_rad', 'windspeed',
                'p_mean', 'pet_mean', 'aridity', 'p_seasonality', 'frac_snow', 'high_prec_freq', 'high_prec_dur',
                'low_prec_freq', 'low_prec_dur', 'benchmark_catch', 'num_reservoir', 'reservoir_cap', 'inter_high_perc',
                'inter_mod_perc', 'inter_low_perc', 'frac_high_perc', 'frac_mod_perc', 'frac_low_perc', 'no_gw_perc',
                'low_nsig_perc', 'nsig_low_perc', 'q_mean', 'runoff_ratio', 'stream_elas', 'baseflow_index',
                'baseflow_index_ceh', 'hfd_mean', 'Q5', 'Q95', 'high_q_freq', 'high_q_dur', 'low_q_freq', 'low_q_dur',
                'zero_q_freq', 'flow_period_start', 'flow_period_end', 'flow_perc_complete', 'quncert_meta',
                'dwood_perc', 'ewood_perc', 'grass_perc', 'shrub_perc', 'crop_perc', 'urban_perc', 'inwater_perc',
                'bares_perc', 'dom_land_cover', 'sand_perc', 'sand_perc_missing', 'silt_perc', 'silt_perc_missing',
                'clay_perc', 'clay_perc_missing', 'organic_perc', 'organic_perc_missing', 'bulkdens',
                'bulkdens_missing', 'bulkdens_5', 'bulkdens_50', 'bulkdens_95', 'tawc', 'tawc_missing', 'tawc_5',
                'tawc_50', 'tawc_95', 'porosity_cosby', 'porosity_cosby_missing', 'porosity_cosby_5',
                'porosity_cosby_50', 'porosity_cosby_95', 'porosity_hypres', 'porosity_hypres_missing',
                'porosity_hypres_5', 'porosity_hypres_50', 'porosity_hypres_95', 'conductivity_cosby',
                'conductivity_cosby_missing', 'conductivity_cosby_5', 'conductivity_cosby_50', 'conductivity_cosby_95',
                'conductivity_hypres', 'conductivity_hypres_missing', 'conductivity_hypres_5', 'conductivity_hypres_50',
                'conductivity_hypres_95', 'root_depth', 'root_depth_missing', 'root_depth_5', 'root_depth_50',
                'root_depth_95', 'soil_depth_pelletier', 'soil_depth_pelletier_missing', 'soil_depth_pelletier_5',
                'soil_depth_pelletier_50', 'soil_depth_pelletier_95', 'gauge_name', 'gauge_lat', 'gauge_lon',
                'gauge_easting', 'gauge_northing', 'gauge_elev', 'area', 'elev_min', 'elev_10', 'elev_50', 'elev_90',
                'elev_max', 'dpsbar', 'elev_mean')
