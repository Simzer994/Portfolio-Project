import numpy as np
import pandas as pd

def get_tickers(path: str):
    """
    Return a list of tickers stored in the file tickers.txt
    """
    with open(path,'r') as f:
        tickers = f.read().splitlines()
    return tickers

def seperate_df(df: pd.DataFrame, begin_train: str, end_train: str, begin_test: str, end_test: str):
    """
    Seperate the datafrale into 2 dataframes,
    one for the train returns and one for the test returns
    """
    train = df.loc[begin_train:end_train]
    test = df.loc[begin_test:end_test]
    return train, test



def portfolio_return(weights,returns):
    """
    Return the portfolio return given the weights and returns
    """
    return (weights.T @ returns)

def portfolio_var(weights,covmat):
    """
    Return the portfolio variance given the weights and covariance matrix
    """
    return (weights.T @ covmat @ weights)

def portfolio_std(weights,covmat):
    """
    Return the portfolio standard deviation given the weights and covariance matrix
    """
    return np.sqrt(portfolio_var(weights,covmat))

