# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:26:17 2020

@author: Matt
"""
import numpy as np
import pickle as pkl

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class stage_predictor:
    """
    Pretty much just contains an instance of the gaussian process regressor 
    from sklearn but with the parameters already set up 
    """
    def __init__(self,
                 kernel_length_scale = 10,
                 kernel_length_bounds = (1e-1,1e3)):
        
        self.kernel = RBF(kernel_length_scale, kernel_length_bounds)
        self.regressor = GaussianProcessRegressor(kernel = self.kernel,
                                                  n_restarts_optimizer=10,
                                                  alpha = 0.1**2)

        
    def train(self, flow_data, stage_data, sample_interval=500):
        """
        Trains the GP stage predictor using samples of the entire dataset
        Because computing K(xi,xj) with 400,000 samples uses lots of memory
        (who would have guessed)
        
        flow_data - entire flow dataset (with NaNs removed)
        stage_data - entire stage dataset (with NaNs removed)
        """
        resh_flow = flow_data.reshape(-1,1)
        
        #Reorder the training data by blow(ascending)
        stage_train_sorted = stage_data[np.argsort(resh_flow, axis=0)] 
        flow_train_sorted = np.sort(resh_flow, axis=0)
        
        #Take every 500th point and 100 at the end to include extreme values
        stage_train_sampled = np.append(stage_train_sorted[0::sample_interval], 
                                        stage_train_sorted[-100:], axis = 0)
        
        flow_train_sampled = np.append(flow_train_sorted[0::sample_interval], 
                                        flow_train_sorted[-100:], axis = 0)
        
        
        self.regressor.fit(flow_train_sampled, stage_train_sampled)
        
    def predict_stage(self, flow):
        """
        Makes stage predictions from flow values, with confidence intervals
        
        flow - list/array of flow values
        """
        if type(flow) == int: #Fix for predicting from integer flow values
            resh_flow = [[flow]]
        else:
            resh_flow = flow.reshape(-1,1)      
        
        y_pred, sigma = self.regressor.predict(resh_flow, return_std=True)       
        return y_pred, sigma
    
    def get_kernel(self):
        return self.regressor.kernel_
    
    def save_model(self, filename):
        pkl.dump(self.regressor, open(filename, 'wb'))
        
    def load_model(self, filename):
        self.regressor = pkl.load(open(filename, 'rb'))
        
        
        
        