from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def extract_mean_value(path, dataset_variable, i, j, k):    	
    if j<10:
        file ='{}_hadukgrid_uk_1km_day_{}0{}01-{}0{}{}.nc'.format(dataset_variable, i, j, i, j, k)
    else:
        file ='{}_hadukgrid_uk_1km_day_{}{}01-{}{}{}.nc'.format(dataset_variable, i, j, i, j, k)
    ds= Dataset(path + file, mode = 'r')
        
    arr = ds[dataset_variable][:][:,::-1,:]
   
    mean_arr = []
    for day in range(k):
        mean_arr.append(np.mean(arr[day][997:1003, 621:641]))
    return mean_arr

path = '/home/ira/Documents/flood_risk_shipston/data/'

cols = ['Year', 'Month', 'Day', 'Catchment Mean Temperature at Surface (Kelvin))', 'Catchment Mean Total Rainfall (mm)']
df = pd.DataFrame(columns = cols)

with open('/home/ira/Documents/flood_risk_shipston/met_office_data_tas_rainfall.csv', "a") as f:
    df.to_csv(f, index=False)

for i in range(2018, 2020):
    for j in range(1, 13):
        if j==2:
            if i%4==0:
                k = 29
            else:
                k = 28
        elif j==4 or j==6 or j==9 or j==11:
            k = 30
        else:
            k = 31
       
        tasmin_arr = extract_mean_value(path, 'tasmin', i, j, k)
        tasmax_arr = extract_mean_value(path, 'tasmax', i, j, k)
        rainfall_arr = extract_mean_value(path, 'rainfall', i, j, k)
       
        for day in range(k):
           
            vals_list = [str(i),
                         str(j),
                         str(day+1),
                         float(273.15+ (tasmin_arr[day]+tasmax_arr[day])/2),
                         float(rainfall_arr[day])]
                             
            print(vals_list)
            df.loc[0] = vals_list
              
            
            with open('/home/ira/Documents/flood_risk_shipston/met_office_data_tas_rainfall.csv', "a") as f:
                df.to_csv(f, index=False, header=False)
       
print(df)


