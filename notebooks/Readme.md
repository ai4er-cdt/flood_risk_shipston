# Notebooks directory
-----

This directory contains notebooks used for exploratory data analysis. Note that the code in the notebooks was not written
with reusability in mind. All reusable source code is in the `src` directory. All finalized analyses from the notebooks are 
in the `reports` directory. 

- `asset_data_exploration.ipynb`: Investigates the cost, installation data and location of flood management interventions in Shipston to understand more 
about the interventions.
- `chess-met-data.ipynb`: Analyses the daily rainfall and wind data from CHESS-MET for the shipston catchment  
- `feh_data_exploration.ipynb`: Analyses the FEH Catchment data of rainfall return periods. We decided not to use this data further after this analysis.
- `flooding_LSTM_first_attempt.ipynb`: First attempt at an LSTM rainfall-runoff model inspired by [Kratzert 2018 et al.](https://doi.org/10.5194/hess-22-6005-2018). Code
for  this first attempt was improved and cleaned and the reusable version is contained in the `src` directory.
- `predictions_analysis_notebook.ipynb`: Analyses the LSTM model's runoff predictions for 2018-2020 versus the ground truth data from the WISKI dataset.
- `runoff_data_collection.ipynb`: Investigates the relationship betwen CAMELS-GB, NRFA and WISKI discharge data and calculates the daily runoff in 
Shipston for the years 1978 to mid 2020.  

The directories `archive` and `mja` contain legacy notebooks that are contained in the notebooks described above.
