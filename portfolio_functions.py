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

def annu_return(df: pd.DataFrame, business_days: int=252):
    """
    Return the annualized returns of a dataframe
    """
    return (df+1).prod()**(business_days/len(df))-1

def asset_vola(df: pd.DataFrame):
    """
    Return the annualized asset volatilities
    """
    return df.std()*(252**0.5)
