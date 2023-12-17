'''
Created on 17 Dec 2023

@author: Big Lion
'''
import pandas as pd

def generate_dummy_data():
    data = {
        'Date': pd.date_range(start='2023-01-01', end='2023-12-31', freq='D'),
        'Market_Price': [100 + i + 10 * (i % 5) + 5 * (i % 10) for i in range(365)]
    }
    return pd.DataFrame(data)

def create_lagged_features(df):
    df['Market_Price_Lag1'] = df['Market_Price'].shift(1)
    df['Market_Price_Lag2'] = df['Market_Price'].shift(2)
    return df.dropna()

