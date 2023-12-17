'''
Created on 17 Dec 2023

@author: Big Lion
'''
import pandas as pd
from market_predictor.data_generator import generate_dummy_data, create_lagged_features
from market_predictor.predictor import train_and_predict

if __name__ == '__main__':
    # Load the dataset
    df = pd.read_csv('dataset.csv')
    
    # Feature engineering: adding lag features
    df_with_lags = create_lagged_features(df)

    # Split data into features (X) and target variable (y)
    X = df_with_lags[['Market_Price_Lag1', 'Market_Price_Lag2']]
    y = df_with_lags['Market_Price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and predict using the predictor module
    predictions = train_and_predict(X_train, y_train, X_test)

    # Plot the results
    plot_results(X_test, y_test, predictions)

