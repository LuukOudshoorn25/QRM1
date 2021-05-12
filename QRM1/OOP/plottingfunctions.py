#################################################
#                                               #    
#           Code for QRM Assignment 1           #
#                 Luuk Oudshoorn                #
#               Willem Jan de Voogd             #
#                                               #    
#################################################

# This file contains functions for the plotting of stuff


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
from helperfunctions import *
#from pypfopt_risk_models import *
import os
import h5py
from joblib import wrap_non_picklable_objects
from hmmlearn import hmm
Ncores=6
#plt.style.use('stylesheet')
plt.style.use('stylesheet')


def plot_timeseries(returns):
    # First plot of all timeseries
    df = np.exp(np.cumsum(returns,axis=0))
    df.plot(lw=0.8)
    plt.legend(frameon=1)
    plt.ylabel('Normalized stock index price')
    plt.axhline(1,ls='--',color='grey',lw=0.5)
    plt.tight_layout()
    plt.savefig('Prices.pdf',bbox_inches='tight')
    plt.show()


def qq_plot(returns, dist='normal',nu=None):
    stocks = returns.columns
    def order(data, sample_size):
        qq = np.ones([sample_size, 2])
        np.random.shuffle(data)
        qq[:, 0] = np.sort(data[0:sample_size])
        qq[:, 1] = np.sort(np.random.normal(size = sample_size))
        if dist == 't':
            qq[:, 1] = np.sort(np.random.standard_t(size = sample_size, df=nu))
        return qq
    
    for stock in stocks:
        measurements = returns[stock]
        qq = order(measurements/np.std(measurements), len(measurements))
        plt.scatter(qq[:,1],qq[:,0],s=0.8,label=stock)
    plt.xlabel('Theoretical quantiles')    
    plt.ylabel('Sample quantiles')
    plt.legend(frameon=1)
    #plt.plot([-6,6],[-6,6],color='black',lw=0.5,ls='--')
    plt.xlim(-5.5,5.5)
    plt.ylim(-5.5,5.5)
    plt.tight_layout()
    plt.savefig('qq_plot'+dist+'.pdf',bbox_inches='tight')
    plt.show()
    #import scipy.stats as stats
    #import statsmodels.api as sm
    #fig,ax=plt.subplots()
    #for res in returns.columns:
    #    sm.qqplot(returns[res], stats.t, distargs=(4,),ax=ax,fit=True, line="45")
    #plt.show()

    
def CDFs(returns):
    stocks = returns.columns
    # Plot CDFs to check normality
    fig,axes = plt.subplots(2,3,sharex=True,sharey=True,figsize=(6,3.4))
    axes = axes.flatten()
    model = hmm.GaussianHMM(n_components=1, covariance_type="full")
    model.fit(returns)
    for i,stock in enumerate(stocks):
        empirical_CDF = ECDF(returns[stock])
        ax = axes[i]
        ax.plot(empirical_CDF.x,empirical_CDF.y,lw=1,color='tomato',label='ECDF')
        x = np.linspace(empirical_CDF.x[1],empirical_CDF.x[-2], 1000)
        fitted_cdf = gauss_cdf(x, model.means_[0,i], np.sqrt(model.covars_[0,i,i]))
        ax.plot(x, fitted_cdf,color='black',lw=0.7,ls='--',label=r'$\mathcal{N}$-CDF')
        ax.set_xlim(-0.07,0.07)
        ax.set_title(stock,y=0.2,x=0.7)
        ax.legend(frameon=1)
    for i in [3,4,5]:
        axes[i].set_xlabel('Return')
    for i in [0,3]:
        axes[i].set_ylabel('CDF')
    #axes[-1].axis('off')
    plt.tight_layout()
    plt.savefig('CDF_comparison.pdf', bbox_inches='tight')
    plt.show()



def EWMA_GARCH(returns):
    # Filtered historic simulation with EWMA
    from arch import arch_model
    fig,axes = plt.subplots(2,3,figsize=(6,4),sharex=True)
    axes=axes.flatten()
    residuals    = pd.DataFrame(index=returns.index)
    volatilities = pd.DataFrame(index=returns.index)
    for i,asset in enumerate(returns.columns):
        ret = returns[asset].dropna()
        dates = ret.index
        ret = ret.values.flatten()
        # Scaling to improve GARCH estimation
        C = 4
        ret = ret * 10**C
        # Fit constant mean GARCH(1,1) model
        am = arch_model(ret, p=1,q=1)
        res = am.fit(update_freq=0,)
        # Get conditional volatilities
        volas = res.conditional_volatility/1e4
        # Estimate empirical distribution of zt
        zt = (ret-ret.mean()) / volas
        axes[i].set_title(returns.columns[i], y=0.8)
        axes[i].plot(dates,np.sqrt(252)*100*volas,color='black',lw=0.5,label='GARCH')
        # EWMA
        EWMA_var = np.zeros(len(ret))
        EWMA_var[:50] = returns[asset].iloc[:50].var()*1e8
        for j in range(50,len(ret)):
            EWMA_var[j] = 0.94 * EWMA_var[j-1] + (1-0.94)*ret[j-1]**2
        EWMA_var = EWMA_var/1e8
        axes[i].plot(dates,np.sqrt(252)*100*np.sqrt(EWMA_var),color='tomato',lw=0.5,label='EWMA')
        #axes[i].set_ylim(0,0.6)
        #axes[i].set_xlim(-8,8)
        #
    axes[0].legend(frameon=1,loc='center left')
    for i in [3,4,5]:
        axes[i].set_xlabel('Date')
        #axes[i].set_xticklabels(axes[i].get_xticklabels(),rotation=45)
        plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45 )
    for i in [0,3]:
        axes[i].set_ylabel('Annualized volatility (%)')
    plt.tight_layout()

    plt.savefig('EWMA_GARCH.pdf',dpi=500)
    plt.show()