# Source code directory:
----

This directory contains all reusable code for the project and is organized as follows:

## 1. Config
Contains all configurations files for the data to use, the model specification and the model training and testing. 
A hierarchical configuration is dynamically generated from the configuration files
via [Hydra](https://hydra.cc/docs/intro/) and [OmegaConf](https://github.com/omry/omegaconf).

## 2. Data
Contains scripts to download the key data needed to run and evaluate the model:
- River discharge data and static basin attributes from the [NRFA API](http://nrfaapps.ceh.ac.uk/nrfa/nrfa-api.html) 
- Long-term records of daily weather variables over the UK from [Chess Met](https://www.ceh.ac.uk/news-and-media/news/chess-meteorological-data-now-available)
- Processing temperature and daily rainfalll data from the [Met Office](https://www.metoffice.gov.uk/). Note: This data is private and has to be requested 
from the Met office directly.

Further data that is needed for model training:
- CAMELS-GB dataset by [Coxon 2020 et al.](https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9) with catchment attributes and hydro-meteorological 
timeseries for 671 catchments across Great Britain. The data can be directly 
downloaded [here](https://catalogue.ceh.ac.uk/documents/8344e4f3-d2ea-44f5-8afa-86d2987543a9).
- WISKI-dataset, a [Shipston](https://en.wikipedia.org/wiki/Shipston-on-Stour) specific river stage dataset for the dates from 2017 onwards, 
which are not covered in CAMELS-GB.  
This data is private and has to be requested from the NRFA. 

## 3. Models
Contains all models used in this project as well as the metrics.   
  
The heart of the models directory is the LSTM model and the training loop for it, which is found in `runoff_model.py`. The model is inspired by 
the [LSTM Rainfall-runoff of Kratzert 2018 et al.](https://hess.copernicus.org/articles/22/6005/2018/). It models the run-off in a catchment on a 
daily basis by taking into account timeseries data on rainfall, temperature, humidity, etc. as well as static catchment attributes. 

Unfortunately, much of the data was not available for the catchment of Shipston, which we focus on in this project. Therefore our final model
is based solely on rainfall and temperature data, as well as static catchment attributes. Nevertheless, the model code is written in a general
way and can directly take into account all features in the CAMELS-GB dataset. 

The `stage_predictor.py` models is a Gaussian process based model that we use connect discharge rates (river flow) to river stages (river height) to 
convert stage data from the Shipston river gauges to discharge data.

## 4. Preprocessing
Contains preprocessing scripts for 
- CAMELS-GB dataset: Creates a Pytorch Dataset from CAMELS-GB that can be used in the LSTM.
- Elevation data: Calculates the average elevation of the catchment via [Google Earth Engine](https://earthengine.google.com/)
- Extreme-value analyser: Extracts extreme values from the WISKI dataset.
