import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def get_tickers(path: str):
    """
    Return a list of tickers stored in the file tickers.txt
    """
    with open(path,'r') as f:
        tickers = f.read().splitlines()
    f.close()
    return tickers

def tickers_storage(tickers: list, path: str):
    """
    Store the tickers in the file tickers.txt
    """
    with open(path,'w') as f:
        f.truncate()
        for ticker in tickers:
            if ticker == tickers[-1]:
                f.write(ticker)
            else:
                f.write(ticker)
                f.write("\n")
    f.close()

def seperate_df(df: pd.DataFrame, begin_train: str, end_train: str, begin_test: str, end_test: str):
    """
    Seperate the datafrale into 2 dataframes,
    one for the train returns and one for the test returns
    """
    train = df.loc[begin_train:end_train]
    test = df.loc[begin_test:end_test]
    return train, test

def daily_asset_return(df: pd.DataFrame):
    """
    Return the daily returns of a dataframe
    """
    return df.pct_change().dropna()

def annualy_asset_return(df: pd.DataFrame, business_days: int=252):
    """
    Return the annualized returns of a dataframe
    """
    return (df+1).prod()**(business_days/df.shape[0])-1

def portfolio_return(weights,rends:pd.DataFrame,show:bool=False):
    """
    Returns the portfolio returns according to the assets considered and the associated weights
    """
    rendements = pd.Series((rends*weights).sum(axis=1),name="Portfolio")
    rendement = annualy_asset_return(rendements)
    if show:
        print("The portfolio return is:",round(rendement*100,2),"%")
    return rendement

def asset_vol(df: pd.DataFrame, business_days: int=252):
    """
    Return the annualized asset volatilities
    """
    vol= df.std()*(business_days**0.5)
    return vol

def portfolio_vol(weights,cov:pd.DataFrame,show:bool=False):
    """
    Returns the portfolio volatility according to the assets considered and the associated weights
    """
    vol = (np.dot(weights.T,np.dot(cov*252,weights)))**0.5
    if show:
        print("The portfolio volatility is:",round(vol*100,2),"%")
    return vol

def gmv_portfolio(rends:pd.DataFrame,cov:pd.DataFrame,show:bool=False):
    """
    Return the annualized return and volatility of the global minimum variance portfolio
    """
    weights = np.dot(np.linalg.inv(cov*252),np.ones(cov.shape[0]))/(np.dot(np.ones(cov.shape[0]),np.dot(np.linalg.inv(cov*252),np.ones(cov.shape[0]))))
    annual_return = portfolio_return(weights,rends,show=show) 
    annual_volatility = portfolio_vol(weights,cov,show=show)
    return weights,annual_return,annual_volatility

def opt_mean_variance(rends:pd.DataFrame,cov:pd.DataFrame,obj_rend:float,risk_free=None,period_per_year:int=252,show:bool=False):
    """
    Renvoie un dictionnaire des poids associés à chaque asset\n
    rends : pd.DataFrame\n
    DataFrame contenant les rendements de chaque assets\n
    cov : pd.DataFrame\n
    Matrice de covariance annualisé des prix\n
    obj_rend : float\n
    Représente le rendement annualisé souhaité\n
    risk_free : float\n
    Représente le rendement annualisé sans risque\n
    period_per_year : int\n
    Période considérée pour calculer la volatilité annualisée\n
    show : bool\n
    Pour afficher ou non le résultat
    """
    if risk_free != None:
        rends = rends.copy()
        rends["Risk Free"] = (1+risk_free)**(1/period_per_year)-1
        cov = rends.cov()
    n_assets = rends.shape[1]
    init_weights = np.repeat(1/n_assets, n_assets)

    # Limites pour les poids
    bounds = ((-1.0, 1.0),) * n_assets

    # Définition des contraintes
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}

    if risk_free == None:
        return_is_target = {'type': 'eq',
                            'args': (rends,),
                            'fun': lambda weights,rends: obj_rend - portfolio_return(weights,rends)}
    else :
        return_is_target = {'type': 'eq',
                            'args': (rends,risk_free,),
                            'fun': lambda weights,rends,risk_free: obj_rend - portfolio_return(weights,rends) - (1-np.sum(weights))*risk_free}
    
    # Fonction d'optimisation
    weights = minimize(portfolio_vol,init_weights,
                       args=(cov,), method='SLSQP',
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)

    w = {rends.columns[i]:round(weights.x[i],3) for i in range(rends.shape[1])}
    r = portfolio_return(weights.x,rends,show=show)
    v = portfolio_vol(weights.x,cov,show=show)
    return w,r,v

def efficient_frontier(rends:pd.DataFrame,cov:pd.DataFrame,min:float,max:float,number:int=25,risk_free=None,plot:bool=False):
    """
    Return two arrays of returns and associate volatility\n
    min : float\n
    Start of the range of returns\n
    max : float\n
    End of the range of returns\n
    number : int\n
    Number of points to consider\n
    plot : bool\n
    To plot or not the efficient frontier
    """
    gmv_w,gmv_r,gmv_vol = gmv_portfolio(rends,cov)
    r_eff,vol_eff = np.linspace(gmv_r,max,number),[]
    eff = []
    if risk_free != None:
        r_eff_rf,vol_eff_rf = np.linspace(risk_free,max,number),[]
        eff_rf = []

    if plot:
        r_eff_low,vol_eff_low = np.linspace(min,gmv_r,number),[]
    for i in range(r_eff.shape[0]):
        w,r,v = opt_mean_variance(rends,cov,r_eff[i])
        vol_eff.append(v)
        eff.append([r_eff[i],v])
        if risk_free != None:
            w,r,v = opt_mean_variance(rends,cov,r_eff_rf[i],risk_free=risk_free)
            vol_eff_rf.append(v)
            eff_rf.append([r_eff_rf[i],v])
        if plot:
            w,r,v = opt_mean_variance(rends,cov,r_eff_low[i])
            vol_eff_low.append(v)
    
    
    if plot:
        plt.plot(vol_eff,r_eff,"black",label="Efficient Frontier")
        if risk_free != None:
            plt.plot(vol_eff_rf,r_eff_rf,"black",label="Capital Market Line",linestyle='dotted')
        plt.plot(vol_eff_low,r_eff_low,"black",label="Low Frontier",linestyle="-.")
        plt.scatter(gmv_vol,gmv_r,color="black",label="GMV Portfolio",marker="o",s=50)
        plt.legend()
        plt.xlabel("Annualized volatility")
        plt.ylabel("Annualized return")
        plt.show()
    
    if risk_free != None:
        return eff,eff_rf
    else:
        return eff