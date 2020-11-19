import numpy as np
import matplotlib.pyplot as plt
import ee

#initalise earth engine (having authorised account:'https://developers.google.com/earth-engine/guides/python_install-conda#get_credentials'
ee.Initialize()

def elevation_mat(x_resolution  = 544, y_resolution = 340, latitude = 52.0607, longitude = -1.6228):
    '''
    Simple function that takes in the:
    resolution: [for x and y directions] (steps) calculated given the grid spacing of the image is 1/13600 degrees.
    latitude and logitude: in this case for Shipston.
    '''

    #ee image for earth elevation: 'https://developers.google.com/earth-engine/datasets/catalog/USGS_SRTMGL1_003'
    ee_map =  ee.Image('USGS/SRTMGL1_003')
    #empty matrix to store values (it is a rectange despite the catchment being a differnt shape. 
    #For a more accurate map of just the catchement a rejection method could be used for those grid spaces not within the catcment.
    mat = np.zeros((x_resolution,y_resolution))

    #loop over the x and y coordinates (ee API is very fast)
    for i, x_val in enumerate(np.linspace(longitude-0.02, longitude+0.02, x_resolution)):
        for j, y_val in enumerate(np.linspace(latitude+0.005, latitude-0.2, y_resolution)): 
            #convert coordinates into ee point locations
            xy = ee.Geometry.Point([x_val, y_val])
            #call the API at the highest resolution (30m) and store in matrix
            mat[i,j] = ee_map.sample(xy, 30).first().get('elevation').getInfo()
    
    #save the output to a csv for analysis
    np.savetxt("elevation_data.csv", mat, delimiter=",")
            
        
