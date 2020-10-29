# Shipston Flood Risk

## Runoff prediction model

This code trains an LSTM deep learning model to predict runoff using a dataset of 671 river basins around the UK.

Code features:

- Automatic real-time logging to the cloud with [wandb](https://wandb.ai/) of model loss, validation metrics, and a plot showing test results, using a publicly available dashboard.
- Full parallelisation across multiple GPUs and multiple nodes.
- Full command line interface to control all major model, dataset, and training options.
    - This uses [hydra](http://hydra.cc/), a framework that allows for composable and type-checked configuration objects created from yaml files.
- PyTorch Lightning model class, allowing for more modular code, less PyTorch boilerplate and easier customisation of the training process.
- Dataset class capable of handling an arbitrary number of river basins, with data from any date range and any number of features.
- Automatic saving of the best k checkpoints according to the validation metric.
- Fully type-hinted and well documented codebase.

### Setup and model training

Before running the model, run `conda env create -f environment.yml` to install all required packages (after installing conda). CUDA 10.1 is required to train on GPU with PyTorch 1.7.

The code is run from `main.py`, the only mandatory command line argument is `run_name`, a short string which describes the run. For example:

`python main.py run_name=test-run`.

An example of a more complex command:

`python main.py run_name=50-epochs-2005 gpus=4 dataset.basins_frac=0.5 dataset.train_test_split=2005 mode.epochs=50 model.dropout_rate=0.2`.


#### Full argument list:
- Main Options:
  - `run_name` - Mandatory string argument that describes the run.
  - `cuda` - Whether to use GPUs for training, defaults to `True`.
  - `gpus` - Number of GPUs to use, defaults to 1.
  - `precision` - Whether to use 32 bit or 16 bit floating points for the model. Warning: 16 bit is buggy. Defaults to 32.
  - `seed` - Random seed, defaults to 42.
  - `parallel_engine` - PyTorch parallelisation algorithm to use. Defaults to `ddp`, meaning DistributedDataParallel.
- Dataset Options
  - `dataset.features` - Dictionary where the keys are feature types and the values are lists of string feature names to use for the model. Defaults to `{'timeseries' : ['precipitation', 'temperature', 'shortwave_rad', 'peti',  'humidity', 'windspeed']}` to use the 6 most useful timeseries features. Full list of features below. Better to change this in `src/configs/config.yaml` rather than the command line.
  - `dataset.seq_length` - Number of previous days of meterological data to use for one prediction, defaults to 365.
  - `dataset.train_test_split` - Split date to separate the data into the train and test sets, defaults to `'2010'` meaning 01-01-2020. You can pass a string in DD-MM-YYYY or YYYY formats.
  - `dataset.basins_frac` - Fraction of basins that will be combined to create the dataset, defaults to 0.1 meaning 10% since the full dataset requires roughly 100 GB of memory.
  - `dataset.shuffle` - Whether to shuffle the training dataset randomly, defaults to `True`.
  - `dataset.num_workers` - Number of subprocesses to use for data loading, defaults to 8.
- Training Options
  - `mode.epochs` - Number of training epochs, defaults to 20.
  - `mode.batch_size` - Size of training batches, defaults to 256.
  - `mode.learning_rate` - Learning rate for the Adam optimiser, defaults to 3e-3.
  - `mode.checkpoint_freq` - How many epochs we should train for before checkpointing the model, defaults to 1.
  - `mode.val_interval` - If this is a float, it is the proportion of the training set that should go between validation epochs. If this is an int, it denotes the number of batches in between validation epochs. Defaults to 0.25, meaning 4 validation epochs per training epoch.
  - `mode.log_steps` - How many gradient updates between each log point, defaults to 50.
  - `mode.date_range` - Custom date range for the training dataset to override the default range of 1970 to `dataset.train_test_split`, as a list of two strings (same formats as `dataset.train_test_split`).
  - `mode.mc_dropout` - Boolean that decides whether or not to use MC Dropout to plot output uncertainty. Defaults to `False`.
  - `mode.mc_dropout_iters` - Number of forward passes to use with MC dropout to get uncertainty, defaults to 20.
- Model Options
  - `model.num_layers` - Number of layers in the LSTM, defaults to 2.
  - `model.bidirectional` - Whether to use Bidirectional LSTM, defaults to `False`.
  - `model.hidden_units` - Number of hidden units/LSTM cells per layer, defaults to 50.
  - `model.dropout_rate` - Dropout probability, where the dropout is applied to the dense layer after the LSTM. Defaults to 0.0


### Dataset and features

The dataset is [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) - the first time it is run the code will automatically download and unzip the dataset to `src/data/CAMELS-GB/` using a Dropbox link.

Each basin has data from 8 different meterological timeseries, as well as many more static basin attributes that have a constant scalar value for the entire basin.

Below is a full list of features that can be included in the model. These are the only features that can be included in the `dataset.features` config option. Full descriptions of all these features can be found in the CAMELS-GB supplementary material.

- Timeseries Features
  - precipitation
  - pet
  - temperature
  - peti
  - humidity
  - shortwave_rad
  - longwave_rad
  - windspeed
- Climatic Features
  - p_mean
  - pet_mean
  - aridity
  - p_seasonality
  - frac_snow
  - high_prec_freq
  - high_prec_dur
  - low_prec_freq
  - low_prec_dur
- Human Influence Features
  - num_reservoir
  - reservoir_cap
- Hydrogeology Features
  - inter_high_perc
  - inter_mod_perc
  - inter_low_perc
  - frac_high_perc
  - frac_mod_perc
  - frac_low_perc
  - no_gw_perc
  - low_nsig_perc
  - nsig_low_perc
- Hydrologic Features
  - q_mean
  - runoff_ratio
  - stream_elas
  - baseflow_index
  - baseflow_index_ceh
  - hfd_mean
  - Q5
  - Q95
  - high_q_freq
  - high_q_dur
  - low_q_freq
  - low_q_dur
  - zero_q_freq
- Land-cover Features
  - dwood_perc
  - ewood_perc
  - grass_perc
  - shrub_perc
  - crop_perc
  - urban_perc
  - inwater_perc
  - bares_perc
  - dom_land_cover
- Topographic Features
  - gauge_lat
  - gauge_lon
  - gauge_easting
  - gauge_northing
  - gauge_elev
  - area
  - elev_min
  - elev_10
  - elev_50
  - elev_90
  - elev_max
  - dpsbar
  - elev_mean
- Soil Features
  - sand_perc
  - sand_perc_missing
  - silt_perc
  - silt_perc_missing
  - clay_perc
  - clay_perc_missing
  - organic_perc
  - organic_perc_missing
  - bulkdens
  - bulkdens_missing
  - bulkdens_5
  - bulkdens_50
  - bulkdens_95
  - tawc
  - tawc_missing
  - tawc_5
  - tawc_50
  - tawc_95
  - porosity_cosby
  - porosity_cosby_missing
  - porosity_cosby_5
  - porosity_cosby_50
  - porosity_cosby_95
  - porosity_hypres
  - porosity_hypres_missing
  - porosity_hypres_5
  - porosity_hypres_50
  - porosity_hypres_95
  - conductivity_cosby
  - conductivity_cosby_missing
  - conductivity_cosby_5
  - conductivity_cosby_50
  - conductivity_cosby_95
  - conductivity_hypres
  - conductivity_hypres_missing
  - conductivity_hypres_5
  - conductivity_hypres_50
  - conductivity_hypres_95
  - root_depth
  - root_depth_missing
  - root_depth_5
  - root_depth_50
  - root_depth_95
  - soil_depth_pelletier
  - soil_depth_pelletier_missing
  - soil_depth_pelletier_5
  - soil_depth_pelletier_50
  - soil_depth_pelletier_95
