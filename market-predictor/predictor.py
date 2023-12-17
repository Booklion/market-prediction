'''
Created on 17 Dec 2023

@author: Big Lion
'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def train_and_predict(X_train, y_train, X_test):
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Make predictions on the test set
    predictions = model.predict(X_test)

    return predictions

def plot_results(X_test, y_test, predictions):
    # Plot the actual vs predicted values
    plt.scatter(X_test['Market_Price_Lag1'], y_test, color='black', label='Actual')
    plt.scatter(X_test['Market_Price_Lag1'], predictions, color='blue', label='Predicted')
    plt.xlabel('Market_Price_Lag1')
    plt.ylabel('Market_Price')
    plt.legend()
    plt.show()
