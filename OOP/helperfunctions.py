#################################################
#                                               #    
#           Code for QRM Assignment 1           #
#                 Luuk Oudshoorn                #
#               Willem Jan de Voogd             #
#                                               #    
#################################################

# This file contains helper functions, like pdfs, cdfs, etc


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal,kstest,norm,t
from scipy.interpolate import interp1d
from scipy.special import erf
from joblib import Parallel, delayed
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from pandas.tseries.offsets import Day, BDay
from datetime import date
import matplotlib.ticker as mtick
#from pypfopt_risk_models import *
import os
import h5py
from joblib import wrap_non_picklable_objects
from hmmlearn import hmm
Ncores=6
#plt.style.use('stylesheet')
plt.style.use('stylesheet')


# Helper functions
# Gaussian PDF
def gauss_pdf(x, mu,var):
    sigma = np.sqrt(var)
    return norm(scale=sigma, loc=mu).pdf(x)

# Gaussian CDF
def gauss_cdf(x,mu,sigma):
    return 0.5*(1+erf((x-mu)/(sigma*np.sqrt(2))))

def gauss_icdf(x,mu,sigma):
    return norm.ppf(x,loc=mu,scale=sigma)

def corr_to_cov(corr_matrix, stdevs):
    """
    Convert a correlation matrix to a covariance matrix

    :param corr_matrix: correlation matrix
    :type corr_matrix: pd.DataFrame
    :param stdevs: vector of standard deviations
    :type stdevs: array-like
    :return: covariance matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(corr_matrix, pd.DataFrame):
        warnings.warn("corr_matrix is not a dataframe", RuntimeWarning)
        corr_matrix = pd.DataFrame(corr_matrix)

    return corr_matrix * np.outer(stdevs, stdevs)


def KS_test(returns, cdf1, cdf2):
    p1 = kstest(returns, cdf1)[1]
    p2 = kstest(returns, cdf2)[1]
    return p1,p2

def load_timeseries():
    # Get filenames using glob
    fnames = glob('./timeseries/*csv')
    for f in fnames:
        newdf = pd.read_csv(f)[['Date','Close']]
        newdf = newdf.set_index('Date')
        newdf.index = pd.to_datetime(newdf.index)
        newdf.columns = [f.split('/')[-1].split('.csv')[0]]
        try:
            df = df.join(newdf)
        except:
            df = newdf
    df = df.dropna()
    return df
    
class dataconversion():
    def __init__(self):
        return
    
    def to_log_ret(self, df):
        """Convert prices to log returns"""
        log_ret = np.log(df/df.shift(1))
        return log_ret.iloc[1:,:]
    
