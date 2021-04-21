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
    

def load_timeseries_new():
    try:
        joined = pd.read_pickle('All_timeseries.pickle')
    except:
        df = pd.read_pickle('./timeseries/indices.pickle')
        # Load US treasuries
        UST5 = pd.read_excel('./timeseries/treasury_5y.xls',header=None)
        UST5.columns = ['Date','Treasury5y']
        UST5.Date = pd.to_datetime(UST5.Date)
        UST5 = UST5.set_index('Date')
        UST5 = UST5.diff()
        # Load AEX
        aex = pd.read_csv('./timeseries/AEX.csv')
        aex['Date'] = pd.to_datetime(aex['Date'])
        aex=  aex.set_index('Date')
        aex = aex[['Close']]
        aex.columns = ['AEX']
        aex = np.log(aex/aex.shift(1)).dropna()
        # Load EUR/USD
        EUR_USD = pd.read_excel('./timeseries/EUR_USD.xls')
        EUR_USD.observation_date = pd.to_datetime(EUR_USD.observation_date)
        EUR_USD = EUR_USD.set_index('observation_date')
        EUR_USD = np.log(EUR_USD/EUR_USD.shift(1))
        # Load EUR/Turkish Lira
        EUR_TRY = pd.read_excel('./timeseries/EUR_TRY.xlsx')
        EUR_TRY.Date = pd.to_datetime(EUR_TRY.Date)
        EUR_TRY = EUR_TRY.set_index('Date')['TRY']
        EUR_TRY = np.log(EUR_TRY/EUR_TRY.shift(1))
        EUR_TRY = EUR_TRY.dropna()
        # Load BIST (turkish stock index)
        bist = pd.read_csv('./timeseries/bist_all_shares.csv')
        bist['Date'] = pd.to_datetime(bist['Date'])
        bist=  bist.set_index('Date')
        bist = bist[['Close']]
        bist.columns = ['Bist']
        bist = np.log(bist/bist.shift(1))
        bist['Bist'].loc['2020-07-27'] = bist['Bist'].mean()
        # Join all
        joined = df.join(EUR_USD).join(EUR_TRY).join(bist).join(UST5).join(aex)
        joined = joined.dropna()
        joined = joined.loc['2011':]
        # Convert to EUR
        joined['Bist EUR'] = joined['Bist'] + joined['TRY']
        joined['Commodities EUR'] = joined['Commodities'] + joined['DEXUSEU']
        joined['US CMBL EUR'] = joined['US CMBL'] + joined['DEXUSEU']
        joined['Nasdaq EUR'] = joined['Nasdaq'] + joined['DEXUSEU']
        joined['Emerging equity EUR'] = joined['Emerging equity'] + joined['DEXUSEU']
        joined['Private Debt EUR'] = joined['Private Debt'] + joined['DEXUSEU']
        joined.to_pickle('All_timeseries.pickle')
    return joined


def test_square_root():
    # Check performance of square root t
    # Step 1: sample 5 day sets of returns
    # Generate sets of 5 random days
    
    weights = np.array([0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1])
    weights = weights/np.sum(weights)
    Ndays = np.arange(5,252,10)
    onedayVaR = -1*np.percentile(np.dot(weights,returns.T),1)
    errs = []
    VaRs = []
    sq_VaRs = []
    for day in Ndays:
        days = np.random.randint(low=0,high=len(returns),size=(50000,day))
        # Get cumulative returns for these days
        portfolio = np.dot(weights,returns.T)
        meanret = np.mean(portfolio)
        portfolio = portfolio[days]
        portfolio = np.sum(portfolio,axis=1)
        # Get empirical VaR
        VaR = -1*np.percentile(portfolio,1)
        sq_root_time_VaR = np.sqrt(day) * onedayVaR
        err = np.abs(VaR - sq_root_time_VaR)
        
        errs.append(err)
        VaRs.append(VaR)
        sq_VaRs.append(sq_root_time_VaR)
        
    plt.figure()
    plt.scatter(Ndays,errs,s=1,label='Absolute error')
    plt.scatter(Ndays,VaRs,s=1,label='Empirical N-day VaR')
    plt.scatter(Ndays,sq_VaRs,s=1,label=r'$\sqrt{N}\times$ VaR')
    plt.legend(loc='best',frameon=1)
    plt.xlabel('Time horizon')
    plt.ylabel('99%-VaR')
    plt.tight_layout()
    plt.savefig('Testing_squareroot_rule.pdf')
    plt.show()


def iterate_over_scenarios(loss_distributions):
    for scenario in ['Drop in equities','Rising yield','Increase in USD','Decrease in TRY', 'Large rise in equities']:
        bins = np.arange(-75,75,1.5)
    plt.hist(loss_distributions[scenario]*100,histtype='step',bins=bins, density=True,label=scenario)
    plt.xlim(-60,60)
    plt.xlabel('Loss (%)')
    plt.ylabel('Density')
    plt.legend(frameon=1,loc='lower right')
    plt.tight_layout()
    plt.savefig('Portfolio_losses_scenarios.pdf')
    plt.show()