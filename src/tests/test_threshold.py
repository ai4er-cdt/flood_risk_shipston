from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages

base = importr('base')
utils = importr('utils')
utils.chooseCRANmirror(ind=1)
utils.install_packages('POT')


from thresholdmodeling import thresh_modeling 
import pandas as pd


def test_thresholding():
    url = 'https://raw.githubusercontent.com/iagolemos1/thresholdmodeling/master/dataset/rain.csv' #saving url
    df =  pd.read_csv(url, error_bad_lines=False) #getting data
    data = df.values.ravel() #turning data into an array
    thresh_modeling.MRL(data, 0.05)
    thresh_modeling.Parameter_Stability_plot(data, 0.05)
    thresh_modeling.gpdfit(data, 30, 'mle')
    thresh_modeling.gpdpdf(data, 30, 'mle', 'sturges', 0.05)
    thresh_modeling.gpdcdf(data, 30, 'mle', 0.05)
    thresh_modeling.qqplot(data,30, 'mle', 0.05)
    thresh_modeling.ppplot(data, 30, 'mle', 0.05)
    thresh_modeling.return_value(data, 30, 0.05, 365, 36500, 'mle')
    thresh_modeling.decluster(data, 30, 30)
