import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.ardl import ARDL
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('bank_data.csv')

# Exploratory Data Analysis (EDA)
# Check for missing values
print(data.isnull().sum())

# Check for data types
print(data.dtypes)

# Check for outliers
print(data.describe(include='all'))

# Check for normality
print(sm.qqplot(data['end_of_day_balance']))
# If the data is not normally distributed, consider transforming it

# Check for stationarity
# ADF test
print(sm.tsa.stattools.adfuller(data['end_of_day_balance']))
# KPSS test
print(sm.tsa.stattools.kpss(data['end_of_day_balance']))

# Check for cointegration between end of day balance, interest rate, and age of the account
print(coint(data['end_of_day_balance'], data['interest_rate'], data['age_of_account']))

# Split the dataset into train and test
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]

# Run the ARDL model
model = ARDL(train_data['end_of_day_balance'], train_data[['interest_rate', 'age_of_account']], maxlag=5).fit()
print(model.summary())

# Evaluate the model
y_hat = model.predict(test_data['end_of_day_balance'])
print(mean_squared_error(test_data['end_of_day_balance'], y_hat))
print(mean_absolute_error(test_data['end_of_day_balance'], y_hat))

# Fine-tune the model
# Consider using different lag lengths
# Consider using different model selection criteria, such as AIC or BIC

# Interpret the impact of independent variables
print(model.coef_)

# Forecast the outcome using the model for the test data set
y_hat_forecast = model.forecast(steps=len(test_data))
print(y_hat_forecast)

# Comprehensive performance statistics for the model
print(model.mse)
print(model.aic)
print(model.bic)

# Visualizations
plt.plot(train_data['end_of_day_balance'], label='Actual')
plt.plot(y_hat, label='Predicted')
plt.legend()
plt.show()

plt.plot(test_data['end_of_day_balance'], label='Actual')
plt.plot(y_hat_forecast, label='Predicted')
plt.legend()
plt.show()
