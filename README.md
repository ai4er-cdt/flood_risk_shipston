# Shipston Flood Risk Project

## 1. Project Summary

#### Abstract
The purpose of this project is to investigate whether we can establish the effectiveness of __natural flood management (NFM)__ [interventions](https://www.therrc.co.uk/sites/default/files/files/Guidance_training/NFM_Roadshow/May_West_Midlands/23may_westmids_shipstonstour_philwraggtomlavers.pdf) undertaken in the British town of [Shipston-on-Stour](https://en.wikipedia.org/wiki/Shipston-on-Stour) during 2017 to 2020 from publicly available meteorological data and private data from the river gauge in Shipston.

Our analysis concludes that the available data (c.f. below for data sources) is not enough to confidently assess the effectiveness of
recent NFM interventions in Shipston with [state-of-the-art rainfall-runoff LSTM models](https://hess.copernicus.org/articles/22/6005/2018/). We attribute this to three main factors:

- __Limited data on extreme events__: The period of 1990 to 2020 contains less than 10 floods in Shipston (ca. 7 independent events if we define a flood by a threshold of 3.4m river stage). This means that while there is ample data on the __average__ discharge in the catchment, there is little information about the __extreme values__ of the river discharge. We see this reflected in the results of our model, as the model's predictions agree well with the ground truth on average dischage values in terms of [Nash-Sutcliff model efficiency (NSE)](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient), but carry errors of about 20-30% for extreme events.
- __Limited temporal resolution of available data__: All publicly available, past meteorological data from the [Met Office](https://www.metoffice.gov.uk/) and [NRFA](https://nrfa.ceh.ac.uk/) is available on a daily basis. Yet, historical data shows that floods in Shipston depend on processes on hourly timescales. Flooding in Shipston strongly depends on whether the peak flow through Shipston exceeds 3.4m (the height of the Shipston bridge arches) at any given time, which typically happens only for a few hours of a day, even during floods. NFM interventions help to __flatten the curve__ and distribute the peak flow across a wider time range by slowing upstream flow rates. Since leakage through NFMs likely occurs on timescales of hours as well (many NFMs are leaky dams),
data with daily temporal resolution is likely not enough to confidently assess the effect of NFM interventions. In other words: NFM interventions might flatten the hourly flow without strongly affecting the daily total flow rate, such that the averaging over all hours in a day removes any information about the NFMs effectiveness.
- __Limited meteorological data availability__: While temperature, precipitation and river discharge data for 1990 to 2020 were readily available, we could not obtain other important meteorological data (notably humidity, windspeed, potential evapotranspiration, or solar irradiation) data for this period for the Shipston catchment. This data is relevant to include information about the physical process of evapotranspiration into the model and its absence means that
our model predictions do not capture all relevant physical mechanisms. We assessed the effect of using only [precipitation and temperature instead of all meteorological data](https://github.com/ai4er-cdt/flood_risk_shipston/issues/17#issuecomment-721790841) via the publicly available [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) dataset and found that we only loose about 2-4% in [NSE](https://en.wikipedia.org/wiki/Nash%E2%80%93Sutcliffe_model_efficiency_coefficient) performance. While this means that evapotranspiration plays a less important role at high latitudes in Britain (Shipston: ~52° N), it is still a significant loss to drainage basins and should be taken into account from a hydrological point of view.

Despite the limits on available data, our [runoff prediction model](https://github.com/ai4er-cdt/flood_risk_shipston/blob/master/README.md#2-runoff-prediction-model) was built in a general way such that it can directly use additional features (e.g. humidity, windspeed, etc.)
contained in the [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) dataset when they become available for Shipston, and it can easily be extended to hourly instead of daily data. Therefore, the model developed in this project can be reused for the effectiveness assessment, once the limits on data-availability have been addressed. A detailed description of how to run the model on the Shipston data or the [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) dataset is given in section 2 of this readme.

### 1.1 Approach
Our approach was to build a model to predict the rainfall-runoff at Shipston in the "what-if" scenario of no NFM interventions being applied. To do this we trained our model on data before 2016 and used it to predict the river flow at Shipston from 2017 to 2020.

For an introduction to using neural networks in hydrology, we refer the interested reader to [this excellent introduction](https://neuralhydrology.github.io/post/research/gauch2020guide/).

#### Data sources
| Data <br>Source 	| Type <br>of Data 	| Temporal <br>resolution 	| Spatial <br>resolution 	| Description 	|
|-	|-	|-	|-	|-	|
| [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) 	| Hydrological data (temperature, <br>precipitation, radiation, humidity, discharge, static attributes, ...) for 671 catchments in the UK 	| daily [1970-2015] 	| lumped 	| Hydrologic attributes and historical forcing data for 671 catchments in the UK. All data in CAMELS-GB is lumped, meaning that the data is a averaged over the catchment area. Unfortunately, the dataset does not include Shipston. We used this data for fine tuning the model as well as for analysing the effect of neglecting evapotranspiration. 	|
| [NRFA Data](https://nrfa.ceh.ac.uk/content/nrfa-api) 	| Discharge data and static <br>catchment attributes for all UK catchments 	| daily <br>[2005-2019] 	| lumped 	| We used this data to compare to CAMELS-GB and Wiski data and to obtain static catchment attributes for Shipston. 	|
| Wiski Data* 	| Precipitation, discharge and stage data for Shipston catchment 	| hourly <br>[1972-2020] 	| lumped 	| We use the hourly precipitation and discharge data for the final model for the Shipston catchment. 	|
| Asset Data* 	| NFM interventions(cost, location, size, type, build date, ...) 	| daily <br>[2017-2020] 	| distributed 	| Data on the NFM assets installed. NFM installationstarted in 2017, hence the data covers the date range from 2017 to 2020. 	|
| [FEH Data](https://fehweb.ceh.ac.uk/) 	| Rainfall return periods 	| - 	| lumped 	| Not used further in our analysis. 	|
| [CHESS-Met Data](https://www.ceh.ac.uk/news-and-media/news/chess-meteorological-data-now-available) 	| Precipitation, temperature for Shipston catchment 	| daily <br>[1970-2020] 	| distributed 	| We used this precipitation and temperature data for the final model on the Shipston catchment. 	|
| [Earth Engine Data](https://earthengine.google.com/) 	| Elevation 	| static 	| distributed 	| We used the elevation data from Google Earth engine to calculate the mean elevation in the catchment. 	|

*Starred data sources are private and could be obtained by direct requests to the respective agencies.

### 1.2 Results

The current model does not seem enough to assess the effectiveness of NFM interventions in this way. A notebook that generates an analysis of the model predictions can be found in `predictions_analysis_notebook.ipynb` in the `notebooks` folder, along with other notebooks analysing separate data from Shipston. This shows that there is no detectable effect from the NFM interventions in the difference between model predictions and ground truth flow at Shipston in the 2017-2020 period, however this is most likely due to the data limitations described above.

### Results for Shipston-only models
This table shows a comparison of the performance of different types of predictive models on the Shipston dataset - the LSTM is clearly the best performing, so this was chosen as the final model.
| Model | Validation NSE (2010-2016) |
|--------|--------------------------------|
| **Tuned Vanilla LSTM**** | **0.8175** |
| [Vanilla 1D Conv model](https://github.com/ai4er-cdt/flood_risk_shipston/blob/425731e381bbc6ad004aed1fb13863bae0026cdb/src/models/conv.py#L108) | 0.4309 |
| [WaveNet](https://arxiv.org/abs/1609.03499) | 0.6975 |
| [FilterNet](https://github.com/Mikata-Project/FilterNet) | 0.5978 |
| Autoregressive* WaveNet | 0.3359 |
| Autoregressive* FilterNet | 0.602 |
| Autoregressive* LSTM | 0.6925 |

*Autoregressive here refers to including the previous 365 days of discharge data as an additional feature.
**Hyperparameters: 10 layers, 100 hidden units, dropout probability of 0.2, 200 epochs of training.

Temperature and precipitation were the baseline features used in all models. The training set consisted of the data from 1986-2010, and validation set was 2010-2016.

### 1.3 Directons for future analysis

The most promising idea for further work is to re-train the model on hourly timeseries data at Shipston, since there is a strong possibility that the effect of the NFM interventions operates on hourly timescales. A larger project would be to fully investigate the transfer learning approach of training a model on CAMELS-GB including static basin attributes, then applying it to Shipston. This is a promising way of improving model performance further.

## 2. Runoff prediction model

This code trains an LSTM deep learning model to predict runoff using data from the Shipston river basin. By default, predictions go up to 2019-12-31 and are saved in the log directory in a subfolder with the run name.

Alternatively, this code can train a model on any number of river basins from the CAMELS-GB dataset of 671 river basins around the UK. See below for customisation options.

Code features:

- Automatic real-time logging to the cloud with [wandb](https://wandb.ai/) of model loss, validation metrics, and a plot showing test results, using a publicly available dashboard.
- Full parallelisation across multiple GPUs and multiple nodes.
- Full command line interface to control all major model, dataset, and training options.
    - This uses [hydra](http://hydra.cc/), a framework that allows for composable and type-checked configuration objects created from yaml files.
- PyTorch Lightning model class, allowing for more modular code, less PyTorch boilerplate and easier customisation of the training process.
- Dataset class capable of handling an arbitrary number of river basins, with data from any date range and any number of features.
- Automatic saving of the best k checkpoints according to the validation metric.
- Fully type-hinted and well documented codebase.

### 2.1 Setup and model training

Before running the model, run `conda env create -f environment.yml` to install all required packages (after installing conda). CUDA 10.1 is required to train on GPU with PyTorch 1.7.

The code is run from `main.py`, the only mandatory command line argument is `run_name`, a short string which describes the run. For example:

`python main.py run_name=test-run`.

An example of a more complex command:

`python main.py run_name=50-epochs-2005 gpus=4 dataset.basins_frac=0.5 dataset.train_test_split=2005 epochs=50 model.dropout_rate=0.2`.

#### Full argument list:
- Main Options:
  - `run_name` - Mandatory string argument that describes the run.
  - `cuda` - Whether to use GPUs for training, defaults to `True`.
  - `gpus` - Number of GPUs to use, defaults to 1.
  - `precision` - Whether to use 32 bit or 16 bit floating points for the model. Warning: 16 bit is buggy. Defaults to 32.
  - `seed` - Random seed, defaults to 42.
  - `parallel_engine` - PyTorch parallelisation algorithm to use. Defaults to `'ddp'`, meaning DistributedDataParallel.
- Dataset Options
  - `dataset` - Can be either `'camels-gb'` or `'shipston'` to choose the dataset. Default is `'shipston'`.
  - `dataset.features` - Dictionary where the keys are feature types and the values are lists of string feature names to use for the model. Defaults to `{'timeseries' : ['precipitation', 'temperature']}` for `dataset=shipston` to use the features for which we have the most data at Shipston. For `dataset=camels-gb`, defaults to `{'timeseries' : ['precipitation', 'temperature', 'windspeed', 'shortwave_rad']}` - additionalCAMELS-GB features can be added from the full list of features below. Better to change this in the config files rather than the command line.
  - `dataset.seq_length` - Number of previous days of meterological data to use for one prediction, defaults to 365.
  - `dataset.train_test_split` - Split date to separate the data into the train and test sets, defaults to `'2010'` meaning 01-01-2020. You can pass a string in DD-MM-YYYY or YYYY formats.
  - `dataset.num_workers` - Number of subprocesses to use for data loading, defaults to 8.
  - `dataset.basins_frac` - Only available for `dataset=camels-gb`, this specifies the fraction of basins that will be combined to create the dataset. Defaults to 0.1 meaning 10% since the full dataset requires roughly 100 GB of memory.
  - `dataset.test_end_date` - Only available for `dataset=shipston`, this specifies the end date of the test dataset, can go up to `'2020'`. Override this value as a string. Defaults to `'2020'`.
- Training Options
  - `epochs` - Number of training epochs, defaults to 200.
  - `batch_size` - Size of training batches, defaults to 256.
  - `learning_rate` - Learning rate for the Adam optimiser, defaults to 3e-3.
  - `checkpoint_freq` - How many epochs we should train for before checkpointing the model, defaults to 1.
  - `val_interval` - If this is a float, it is the proportion of the training set that should go between validation epochs. If this is an int, it denotes the number of batches in between validation epochs. Defaults to 0.25, meaning 4 validation epochs per training epoch.
  - `log_steps` - How many gradient updates between each log point, defaults to 20.
  - `date_range` - Custom date range for the training dataset to override the default range of 1970 to `dataset.train_test_split`, as a list of two strings (same formats as `dataset.train_test_split`).
  - `mc_dropout` - Boolean that decides whether or not to use MC Dropout to plot output uncertainty. Defaults to `False`.
  - `mc_dropout_iters` - Number of forward passes to use with MC dropout to get uncertainty, defaults to 20. Increase up to 100 to tighten uncertainty bounds but be aware this can produce memory errors.
- Model Options
  - `model` - Can be either `'lstm'` or `'conv'` to choose between the LSTM or 1D convolutional model. Default is `'lstm'`.
  - LSTM Model Options:
    - `model.num_layers` - Number of layers in the LSTM, defaults to 10.
    - `model.hidden_units` - Number of hidden units/LSTM cells per layer, defaults to 100.
    - `model.dropout_rate` - Dropout probability, where the dropout is applied to the dense layer after the LSTM. Defaults to 0.2
  - Convolutional Model Options:
    - `model.wavenet_kernel_size` - Size of the convolution kernel in the WaveNet model, defaults to 3.


### 2.2 Dataset and features

#### Shipston dataset

The default dataset consists of average daily temperature, precipitation (mm/day), and catchment discharge (m^3/s) time series from 1985 to the end of 2019 for the Shipston river basin. By default the model trains on 1985-2010 and predicts daily average discharge volume for 2010-2019.

#### CAMELS-GB dataset

The [CAMELS-GB](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9) dataset will also automatically download to `src/data/CAMELS-GB/` using a Dropbox link if it is chosen using the command line option. Each basin in CAMELS-GB has data from 8 different meterological timeseries, as well as many more static basin attributes that have a constant scalar value for the entire basin.

Below is a full list of features that can be included in the model with this dataset. These are the only features that can be included in the `dataset.features` config option. Full descriptions of all these features can be found in the CAMELS-GB supplementary material.

- Timeseries Features (daily averages)
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
