#################################################
#                                               #    
#           Code for QRM Assignment 1           #
#                 Luuk Oudshoorn                #
#               Willem Jan de Voogd             #
#                                               #    
#################################################

# Import some of the required libraries
from operator import sub
from numpy.core.numeric import roll
import pandas as pd
import numpy as np
from arch import arch_model
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from glob import glob
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm
from scipy.special import erf
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from scipy.stats import multivariate_normal,kstest,norm,t
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, skew
from scipy.special import erf
from joblib import Parallel, delayed
from scipy.optimize import minimize
from statsmodels.distributions.empirical_distribution import ECDF
from pandas.tseries.offsets import Day, BDay
from datetime import date
import matplotlib.ticker as mtick
import os
import h5py
from joblib import wrap_non_picklable_objects
from hmmlearn import hmm
# Load our own helper and plottingfunctions
from plottingfunctions import *
from helperfunctions import *
# Set number of cores if we do something parallel
Ncores=6
# Set stylesheet for nice plotting
plt.style.use('stylesheet')

# Start of actual code...

returns = load_timeseries_new()
stocks = ['Europe Equity',
          'France All Maturities',
          'AEX',
          'Bist EUR',
          'Commodities EUR',
          'US CMBL EUR',
          'Nasdaq EUR',
          'Emerging equity EUR']
# Make summary table
summary_df = pd.DataFrame({'Annualized mean return':returns.mean()*252*100,
                           'Annualized volatility': returns.std()*np.sqrt(252)*100,
                           'Kurtosis':kurtosis(returns),
                           'Skewness':skew(returns)})
print(summary_df.round(1).sort_index().to_latex(bold_rows=True))

# Plot
plot_timeseries(returns[stocks])
# Do some testing for normality assumptions

# Make QQ plot
qq_plot(returns[stocks], dist='normal')
qq_plot(returns[stocks], dist='t',nu=4.5)
# Make CDF plots
CDFs(returns[stocks[2:]])
# Make GARCH and EWMA plot
EWMA_GARCH(returns[stocks[2:]])

# Do the actual VaR and ES estimation 
class VaR_Predictor():
    def __init__(self,returns):
        """Initialization of VaR Prediction class
            Output of set of 8 year VaR and ES predictions 
            using different techniques"""
        # Set returns as attribute such that we can use it througout the class
        self.returns = returns
        self.stocks  = stocks
        #self.weights = np.array([0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1])
        #self.weights = self.weights/np.sum(self.weights)
        self.weights = np.ones(8)/8
        self.stressed = self.__stressed_periods()

    def __portfolio_returns__(self):
        """Get returns of the portfolio instead of individual assets"""
        portfolio = pd.Series(np.dot(self.weights,self.returns.T),index=self.returns.index)
        return portfolio

    def __portfolio_variance__(self, covmat):
        """Get portfolio variance given weights and covariance matrix"""
        return np.dot(np.dot(self.weights.T, covmat), self.weights)

    def __stressed_periods(self, threshold_percentile=90):
        """Find stressed periods using quantile based regression
           on volatilities"""
        # Estimate the rolling volatility for an equally weighted portfolio
        portfolio = self.__portfolio_returns__()
        rolling_variance = portfolio.rolling(15).var()
        # Set threshold
        threshold = np.percentile(rolling_variance.dropna(), threshold_percentile)
        # Get all days where we are above the threshold
        stressed = rolling_variance>threshold
        # Return dates where we are in a stressed period
        return stressed.index[stressed]

    def __var_covar__(self,exclude_stressed=False,estimation_period=2*252,covariance_method='LediotWolf',theta=0.96,studentt_dof = None):
        """Apply variance covariance matrix method, with given
           estimation period (standard equal to 2 years) and 
           boolean operator to include / exclude stressed periods
           Covariance method can be LedoitWolf or normal and we can
           either pass a value for theta to iteratively update the 
           covariance matrix (see book) or we use a stationary covariance
           matrix in each of the windows. 
            """
        # We iterate over time windows
        predictions = pd.DataFrame({'VaR01' :np.zeros(len(self.returns))*np.nan,
                                    'VaR025':np.zeros(len(self.returns))*np.nan,
                                    'ES01'  :np.zeros(len(self.returns))*np.nan,
                                    'Date'  :self.returns.index}).set_index('Date')
        if theta:
            covmats = []
        if theta==None:
            # Use trailing window covariance matrices
            for iloc1 in range(0,len(self.returns)-estimation_period):
                # Get the returns
                subreturns = self.returns.iloc[iloc1:iloc1+estimation_period]
                # Do we want to exclude stressed periods?
                if exclude_stressed:
                    subreturns = subreturns.drop([w for w in self.stressed if w in subreturns.index])
                # Estimate covariance matrix
                if covariance_method == 'LedoitWolf':
                    covmat = LedoitWolf().fit(subreturns)
                else:
                    covmat = subreturns.cov()
                # Get VaRs and ES estimates for the next day, (so semi-unconditional)
                portfolio_variance   = self.__portfolio_variance__(covmat)
                portfolio_volatility = np.sqrt(portfolio_variance)
                if studentt_dof == None:
                    # Use normal distribution
                    VaR01  = -1*np.dot(self.weights,subreturns.mean().values) + norm.ppf(1-0.01)*portfolio_volatility
                    VaR025 = -1*np.dot(self.weights,subreturns.mean().values) + norm.ppf(1-0.025)*portfolio_volatility
                    ES01   = -1*np.dot(self.weights,subreturns.mean().values) + 1/0.01 * norm.pdf(norm.ppf(0.025))*portfolio_volatility
                else:
                    # Use student-t distribution
                    VaR01  = -1*np.dot(self.weights,subreturns.mean().values) + norm.ppf(1-0.01)*portfolio_volatility * np.sqrt(studentt_dof / (studentt_dof-2))
                    VaR025 = -1*np.dot(self.weights,subreturns.mean().values) + norm.ppf(1-0.025)*portfolio_volatility * np.sqrt(studentt_dof / (studentt_dof-2))
                    ES01   = -1*np.dot(self.weights,subreturns.mean().values) + 1/0.01 * norm.pdf(norm.ppf(0.025))*portfolio_volatility * np.sqrt(studentt_dof / (studentt_dof-2))
                # Prediction data
                pred_date = self.returns.iloc[iloc1+estimation_period:iloc1+estimation_period+2].index[-1]
                predictions.loc[pred_date] = [VaR01,VaR025,ES01]

        if theta:
            # Use theta updating method
            # Get initial covariance matrix
            subreturns = self.returns.iloc[0:estimation_period]
            # Do we want to exclude stressed periods?
            if exclude_stressed:
                subreturns = subreturns.drop([w for w in self.stressed if w in subreturns.index])
            init_covmat = subreturns.cov()
            covmats.append(init_covmat)

            for i,(date, X) in enumerate(self.returns.iloc[estimation_period:].iterrows()):
                X = X.values
                X = X - self.returns.mean().values
                X = X.reshape(-1,1)
                covmats.append(theta * covmats[i] + (1-theta) * np.dot(X,X.T))
                portfolio_variance   = self.__portfolio_variance__(covmats[-1])
                portfolio_volatility = np.sqrt(portfolio_variance)
                VaR01  = -1*np.dot(self.weights,self.returns.mean().values) + norm.ppf(1-0.01)*portfolio_volatility
                VaR025 = -1*np.dot(self.weights,self.returns.mean().values) + norm.ppf(1-0.025)*portfolio_volatility
                ES01   = -1*np.dot(self.weights,self.returns.mean().values) + 1/0.01 * norm.pdf(norm.ppf(0.025))*portfolio_volatility
                # Prediction data
                pred_date = self.returns.iloc[estimation_period+i:i+estimation_period+2].index[-1]
                predictions.loc[pred_date] = [VaR01,VaR025,ES01]
        return predictions


    def __FHS__(self, ewma_lambda = 0.94, timeframe = 2*252):
        predictions = pd.DataFrame({'VaR01' :np.zeros(len(self.returns))*np.nan,
                                    'VaR025':np.zeros(len(self.returns))*np.nan,
                                    'ES01'  :np.zeros(len(self.returns))*np.nan,
                                    'Date'  :self.returns.index}).set_index('Date')
        for time_iter in range(0,len(self.returns)-timeframe):
            # Get data
            subreturns = self.returns.iloc[time_iter:time_iter+timeframe]
            residuals    = pd.DataFrame(index=subreturns.index)
            sigma_preds = {}
            for i,asset in enumerate(self.stocks):
                ret = subreturns[asset].dropna()
                dates = ret.index
                ret = ret.values.flatten()
                # Estimate EWMA model
                EWMA_var = np.zeros(len(ret))
                EWMA_var[:50] = np.var(ret[:50])
                for j in range(50,len(ret)):
                    EWMA_var[j] = ewma_lambda * EWMA_var[j-1] + (1-ewma_lambda)*ret[j-1]**2
                EWMA_vol = np.sqrt(EWMA_var)
                # Predict volatility one day ahead
                sigma_pred = np.sqrt(ewma_lambda*EWMA_var[-1] + (1-ewma_lambda)*ret[-1]**2)
                sigma_preds[asset] = sigma_pred
                # Estimate empirical distribution of zt
                zt = (ret-ret.mean()) / EWMA_vol
                # Save residuals
                residuals.loc[dates,asset] = zt
            # Simulate returns for day t+1
            simulated_returns = residuals* sigma_preds
            # Get portfolio returns
            portfolio = np.dot(self.weights,simulated_returns.T)
            
            VaR01  = -1*np.percentile(portfolio,1)
            VaR025 = -1*np.percentile(portfolio,2.5)
            ES01   = -1*np.mean(portfolio[portfolio<-1*VaR01])

            # Prediction data
            pred_date = self.returns.iloc[time_iter:time_iter+timeframe+1].index[-1]
            predictions.loc[pred_date] = [VaR01,VaR025,ES01]
        return predictions


    def __CCC__(self, estimation_period=2*252, exclude_stressed=False):
        """Dynamic Constant Correlation method.
           Estimate GARCH(1,1) for each asset, estimate correlation matrix and obtain covariance matrix by using
           volatilities sigma from GARCH and correlations from fixed correlation matrix
           Filtered historic simulation with EWMA"""
        predictions = pd.DataFrame({'VaR01' :np.zeros(len(self.returns))*np.nan,
                                    'VaR025':np.zeros(len(self.returns))*np.nan,
                                    'ES01'  :np.zeros(len(self.returns))*np.nan,
                                    'Date'  :self.returns.index}).set_index('Date')

        # Estimate constant correlation matrix using GARCH
        subreturns = self.returns
        # Do we want to exclude stressed periods?
        if exclude_stressed:
            subreturns = subreturns.drop([w for w in self.stressed if w in subreturns.index])
        residuals    = pd.DataFrame(index=subreturns.index)
        volatilities = pd.DataFrame(index=subreturns.index)
        fitted_models = {}
        for i,asset in enumerate(self.stocks):
            ret = subreturns[asset].dropna()
            dates = ret.index
            ret = ret.values.flatten()
            # Scaling to improve GARCH estimation
            C = 3
            ret = ret * 10**C
            # Fit constant mean GARCH(1,1) model
            am = arch_model(ret, p=1,q=1)
            res = am.fit(update_freq=0,)
            # Get conditional volatilities
            volas = res.conditional_volatility
            # Estimate empirical distribution of zt
            zt = (ret-ret.mean()) / volas
            residuals.loc[dates,asset] = zt
            volatilities.loc[dates,asset] = volas
            fitted_models[asset] = res

        # Estimate fixed correlation matrix
        cons_corrmat = residuals.corr()
        # Use volatilities from GARCH to estimate VaRs
        for date,returns in self.returns.iloc[:-1].iterrows():
            # predict volatility one day ahead
            omegas = np.array([fitted_models[w].params['omega'] for w in self.stocks])
            alphas = np.array([fitted_models[w].params['alpha[1]'] for w in self.stocks])
            betas = np.array([fitted_models[w].params['beta[1]'] for w in self.stocks])
            pred_vola = np.sqrt(omegas+alphas*(returns* 10**C)**2+betas*volatilities.loc[date]**2) / (10**C)
            CCC_covmat = corr_to_cov(cons_corrmat, pred_vola.values)
            portfolio_variance = self.__portfolio_variance__(CCC_covmat)
            
            VaR01  = np.sqrt(portfolio_variance) * norm.ppf(1-0.01) - np.dot(self.weights, self.returns.mean())
            VaR025 = np.sqrt(portfolio_variance) * norm.ppf(1-0.025) - np.dot(self.weights, self.returns.mean())
            ES01   = np.sqrt(portfolio_variance) * 1/0.01 * norm.pdf(norm.ppf(0.01)) - np.dot(self.weights, self.returns.mean())
            # Prediction data
            pred_date = self.returns.loc[date:].index[1]
            predictions.loc[pred_date] = [VaR01,VaR025,ES01]
        return predictions

    def __historic_simulation__(self, timeframe = 2*252):
        """Simple historic simulation using different lookback windows"""
        predictions = pd.DataFrame({'VaR01' :np.zeros(len(self.returns))*np.nan,
                                    'VaR025':np.zeros(len(self.returns))*np.nan,
                                    'ES01'  :np.zeros(len(self.returns))*np.nan,
                                    'Date'  :self.returns.index}).set_index('Date')
        for time_iter in range(0,len(self.returns)-timeframe):
            # Get data
            subreturns = self.returns.iloc[time_iter:time_iter+timeframe]
            # Sample returns from this
            portfolio = np.dot(self.weights, subreturns.T)
            VaR01  = -1*np.percentile(portfolio,1)
            VaR025 = -1*np.percentile(portfolio,2.5)
            ES01   = -1*np.mean(portfolio[portfolio<-1*VaR01])

            # Prediction data
            pred_date = self.returns.iloc[time_iter:time_iter+timeframe+1].index[-1]
            predictions.loc[pred_date] = [VaR01,VaR025,ES01]
        return predictions


    def __combine__(self):
        # Variance - covariance method
        output = {}
        output['VarCovar Ledoit Wolf including stressed'] = self.__var_covar__(theta=None)
        output['VarCovar theta including stressed'] = self.__var_covar__()
        output['VarCovar Ledoit Wolf excluding stressed'] = self.__var_covar__(theta=None, exclude_stressed=True)
        output['VarCovar Normal Covmat including stressed'] = self.__var_covar__(theta=None, covariance_method='normal')
        output['VarCovar student 3 including stressed'] = self.__var_covar__(theta=None, covariance_method='normal', studentt_dof=3)
        output['VarCovar student 4 including stressed'] = self.__var_covar__(theta=None, covariance_method='normal', studentt_dof=4)
        output['VarCovar student 4 excluding stressed'] = self.__var_covar__(theta=None, covariance_method='normal', studentt_dof=4, exclude_stressed=True)
        output['VarCovar student 5 including stressed'] = self.__var_covar__(theta=None, covariance_method='normal', studentt_dof=5)
        output['VarCovar student 6 including stressed'] = self.__var_covar__(theta=None, covariance_method='normal', studentt_dof=6)
        output['Historic simulation'] = self.__historic_simulation__()
        output['VarCovar Normal Covmat excluding stressed'] = self.__var_covar__(theta=None, covariance_method='normal', exclude_stressed=True)
        output['FHS'] = self.__FHS__()
        output['Constant correlation method including stressed period'] = self.__CCC__(exclude_stressed=False)
        #output['Constant correlation method excluding stressed period'] = self.__CCC__(exclude_stressed=False)
        return output


VaRP = VaR_Predictor(returns[stocks])
# Get set of different estimated VaR/ES
dictionary = VaRP.__combine__()

class VaR_Analyzer():
    def __init__(self, dictionary, returns):
        """class to analyze the performance of the VaR models developed in the VaR_predictor class"""
        # Make dictionary and returns attributed
        self.dictionary = dictionary
        self.returns = returns
        self.weights = VaRP.weights
        # Join data
        self.join_data()


    def join_data(self):
        # Loop over models
        all_VaR01  = pd.DataFrame({'Date'  :self.returns.index}).set_index('Date')        
        all_VaR025 = pd.DataFrame({'Date'  :self.returns.index}).set_index('Date')        
        all_ES01   = pd.DataFrame({'Date'  :self.returns.index}).set_index('Date')        
        for key in self.dictionary.keys():   
            # Join the data 
            all_VaR01 = pd.merge(all_VaR01,self.dictionary[key]['VaR01'], left_index=True,right_index=True)
            all_VaR025 = pd.merge(all_VaR025,self.dictionary[key]['VaR025'], left_index=True,right_index=True)
            all_ES01 = pd.merge(all_ES01,self.dictionary[key]['ES01'], left_index=True,right_index=True)

        # Give names to the columns equal to the models    
        all_VaR01.columns = [w for w in self.dictionary.keys()]
        all_VaR025.columns = [w for w in self.dictionary.keys()]
        all_ES01.columns = [w for w in self.dictionary.keys()]
        

        # Now add to the VaR models the actual (negative) returns = losses
        portfolio_losses = pd.DataFrame({'True losses':-1*np.dot(self.weights, self.returns.T)}, index=self.returns.index)
        self.all_VaR01 = pd.merge(all_VaR01, portfolio_losses, left_index=True,right_index=True).dropna()
        self.all_VaR025 = pd.merge(all_VaR025, portfolio_losses, left_index=True,right_index=True).dropna()
        self.all_ES01 = pd.merge(all_ES01, portfolio_losses, left_index=True,right_index=True).dropna()
        print(self.all_VaR01)

    def __score_VaR__(self):
        """Function to score the VaR models"""
        # Count actual number of VaR exceedings
        # Iterate over models
        results = pd.DataFrame({'VaR01 exceedings (#)':[], 'VaR01 exceedings (%)':[], 'VaR 2.5 exceedings (#)':[], 'VaR 2.5 exceedings (%)':[], 'Model':[]}).set_index('Model')
        fig, ax = plt.subplots(figsize=(7,2.8))
        ax.plot(100*self.all_VaR01['True losses'],lw=0.4,color='black')
        for model in self.all_VaR01.columns[:-1]:
            exceedings01 = (self.all_VaR01['True losses'] >= self.all_VaR01[model]).sum()
            fraction_exceedings01 = (self.all_VaR01['True losses'] >= self.all_VaR01[model]).mean()*100

            exceedings025 = (self.all_VaR025['True losses'] >= self.all_VaR025[model]).sum()
            fraction_exceedings025 = (self.all_VaR025['True losses'] >= self.all_VaR025[model]).mean()*100
            results.loc[model] = [exceedings01, fraction_exceedings01, exceedings025, fraction_exceedings025]
            ax.plot(100*self.all_VaR01[model],label=model,lw=0.8)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.ylabel('Loss (VaR) [%]')
        plt.xlabel('Date')
        plt.ylim(-0.02*100,100*0.12)
        plt.xlim(pd.to_datetime('2013',format='%Y'),pd.to_datetime('2028',format='%Y'))
        plt.legend(loc='center right',frameon=1,fontsize=7.7)
        plt.savefig('./all_VaR.png', bbox_inches='tight', dpi=500)
        plt.show()
        print(results)

        # Now do hypothesis testing
        num_obs = len(self.all_VaR01)
        expected_01  = 0.01*num_obs
        expected_025 = 0.025*num_obs
        t_stat01  = (results['VaR01 exceedings (#)'] - expected_01) / np.sqrt(num_obs*0.01*0.99)
        t_stat025 = (results['VaR 2.5 exceedings (#)'] - expected_025) / np.sqrt(num_obs*0.025*0.975)
        # Obtain p values for H0: they are the same
        pval01 = np.round(1-norm.cdf(t_stat01),2)
        pval025 = np.round(1-norm.cdf(t_stat025),2)
        # Place the pvalues in the table
        results['VaR01 exceedings (#)'] = results['VaR01 exceedings (#)'].astype(str) + [' ('+str(w)+')' for w in pval01]
        results['VaR 2.5 exceedings (#)'] = results['VaR 2.5 exceedings (#)'].astype(str) + [' ('+str(w)+')' for w in pval025]
        print(results)
        return results

    def __score_ES__(self):
        """Function to score the ES estimations"""
        # Count actual number of VaR exceedings
        # Iterate over models
        results = pd.DataFrame({'True ES01':[], 'Expected ES01':[], 't stat difference':[], 'p-value':[], 'Model':[]}).set_index('Model')
        #fig, ax = plt.subplots()
        #ax.plot(self.all_VaR01['True losses'],lw=0.2,color='black')
        for model in self.all_ES01.columns[:-1]:
            losses = self.all_ES01['True losses']
            TrueVaR01 = np.percentile(losses,99)
            TrueES01 = np.mean(losses[losses>TrueVaR01])
            std_ES01 = np.std(losses[losses>TrueVaR01])
            ES_hat = self.all_ES01[model].mean()
            #ES_std = self.all_ES01[model].std()
            ES_N   = self.all_ES01[model].count()
            # Hypothesis testing
            t_stat = np.abs(TrueES01-ES_hat) / std_ES01
            pval = np.round(1-norm.cdf(t_stat),2)
            results.loc[model] = [TrueES01, ES_hat, t_stat,pval]
        print(results)
        return results



VaRA = VaR_Analyzer(dictionary,returns[stocks])
VaR_results = VaRA.__score_VaR__()
ES_results  = VaRA.__score_ES__()

print(VaR_results.round(3).to_latex(bold_rows=True)) 
print(ES_results.to_latex(bold_rows=True)) 


##################################################
#                 Stress testing                 #
##################################################

class stress():
    def __init__(self, returns):
        """Init function for stress tester. Call with returns dataframe. Set number of simulations and the number of paths per scenario"""
        self.returns = returns
        Nsims = 2000000
        self.Npaths = 5000
        # Sample random draws
        days = np.random.randint(low=0,high=len(returns),size=(Nsims,252))
        self.days = days
        # Get paths
        self.__rvs__()
        # Set weights
        # Note, now not 8 weights, but all since we take complete dataframe with us
        # that is needed to get correct correlation structure
        weights = np.array([0,1,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1])
        self.weights = weights/np.sum(weights)


    def __rvs__(self):
        """Sample from all risk factors the days that we randomly generated. Save as attributed for quick access"""
        # Sample returns from all indices needed for shocks
        self.rvs = {'Nasdaq':self.returns['Nasdaq'].values[self.days],
                    'USD'   :self.returns['DEXUSEU'].values[self.days],
                    'TRY'   :self.returns['TRY'].values[self.days],
                    'Commos':self.returns['Commodities'].values[self.days],
                    'Yield':self.returns['Treasury5y'].values[self.days]}                 
        # Get terminal values in the simulations
        # Note: exponents for indices, commodities and FX, sum for rates!
        self.terminal = {'Nasdaq':np.exp(np.sum(self.rvs['Nasdaq'],axis=1))-1,
                         'USD'   :np.exp(np.sum(self.rvs['USD'],axis=1))-1,
                         'TRY'   :np.exp(np.sum(self.rvs['TRY'],axis=1))-1,
                         'Commos':np.exp(np.sum(self.rvs['Commos'],axis=1))-1,
                         'Yield':np.sum(self.rvs['Yield'],axis=1)}


    def __scenarios__(self,riskfactor,change):
        """risk factor can be 
        Nasdaq
        USD
        TRY
        Commos
        Yield
        Change should be fraction loss (eg -0.2 for 20% loss) of change for yield (eg 2 for 2% increase)
        """
        scenarios = np.argsort(np.abs(self.terminal[riskfactor]-change))[:self.Npaths]
        return scenarios

    def __portfolio_loss__(self,scenarios, plot=False,fname=None):
        """Get the portfolio loss distribution in a particular scenario"""
        # Define array for portfolio values
        portfolios = np.zeros((len(self.weights),*self.days[scenarios].shape))
        # Iterate over risk factors and save returns
        for i,risk_factor in enumerate(self.returns.columns):
            returns = self.returns[risk_factor].values[self.days][scenarios]
            portfolios[i,:,:] = returns
        # Get portfolio returns by taking dot product with weights (ie weighted average of returns)
        portfolio = np.dot(self.weights, portfolios.swapaxes(1,0))
        # Equate cumulative return
        portfolio = np.exp(np.sum(portfolio,axis=1))-1
        # Loss is minus gain
        portfolio_loss = -portfolio
        if plot:
            bins = np.arange(-75,75,1.5)
            plt.hist(100*portfolio_loss,label='Portfolio loss',histtype='step',bins=bins, density=True,)
            plt.xlim(-80,105)
            plt.xlabel('Loss (%)')
            plt.legend(frameon=1,loc='upper right')
            plt.tight_layout()
            if fname:
                plt.savefig(fname)
            plt.show()
        # Get VaR

        VaR01 = np.percentile(portfolio_loss,99)
        VaR025 = np.percentile(portfolio_loss,97.5)
        ES01 = np.mean(portfolio_loss[portfolio_loss>VaR01])
        return VaR01,VaR025, ES01, portfolio_loss
    
    def __conditionals__(self,scenarios,fname=None):
        """Plot conditional loss distribution functions for all assets"""
        bins = np.arange(-75,75,1.5)
        for stock in stocks:
            plt.hist(100-100*np.exp(np.sum(self.returns[stock].values[self.days][scenarios],axis=1)),histtype='step',bins=bins, density=True,label=stock)
        plt.xlim(-80,105)
        plt.xlabel('Loss (%)')
        plt.legend(frameon=1,loc='upper right')
        plt.tight_layout()
        if fname:
            plt.savefig(fname)
        plt.show()
    

    

MyStress = stress(returns)
# Define a dataframe to store the results
VaR_ES = pd.DataFrame({'Scenario':[],'VaR01':[], 'VaR025':[], 'ES01':[], 'Mean loss (gain)':[]}).set_index('Scenario')
# This dictionary will contain the loss distribtuions of all scenarios
loss_distributions = {}
# Iterate over the scenarios
for risk_driver, change, scenario_name in  [('Nasdaq',-0.2, 'Drop in equities'), # Define by hand all scenarios!
                                            ('Nasdaq',0.2, 'Rise in equities'),
                                            ('Nasdaq',-0.4, 'Large drop in equities'),
                                            ('Nasdaq',0.4, 'Large rise in equities'),
                                            ('USD',0.1, 'Increase in USD'),
                                            ('USD',-0.1, 'Decrease in USD'),
                                            ('TRY',0.2, 'Increase in TRY'),
                                            ('TRY',-0.2, 'Decrease in TRY'),
                                            ('Commos',0.2, 'Increase in commos'),
                                            ('Commos',-0.2, 'Decrease in commos'),
                                            ('Yield',2, 'Rising yield'),
                                            ('Yield',-2, 'Decreasing yield')]:
    scenarios = MyStress.__scenarios__(risk_driver,change)
    # Get resulting VaR, ES and loss distributions
    VaR01,VaR025,ES01,loss = MyStress.__portfolio_loss__(scenarios)
    VaR_ES.loc[scenario_name] = [VaR01,VaR025,ES01, np.mean(loss)]
    # Save loss distribution
    loss_distributions  [scenario_name] = loss
    print('Done with ',scenario_name)

# Get VaR in base scenario (ie random draw)
base_scenario = np.random.randint(0,2000000,10000)
VaR01,VaR025,ES01,loss = MyStress.__portfolio_loss__(base_scenario)
print('Base scenario has VaR01 of ',VaR01)
# Plot some of the loss distributions
iterate_over_scenarios(loss_distributions)



print(VaR_ES.round(3).to_latex(bold_rows=True)) 

##################################################
#               Square root testing              #
##################################################
test_square_root()