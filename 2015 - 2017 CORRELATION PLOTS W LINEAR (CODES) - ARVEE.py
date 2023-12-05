#!/usr/bin/env python
# coding: utf-8

# In[1]:


#2015 - 2017
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('CORR15-17.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Select the two variables to calculate correlation and create a new dataframe
X = data['GNSS-DERIVED PWV']
y = data['MODIS-PWV(mm)']
df = pd.concat([X, y], axis=1)

# Calculate the correlation coefficient
corr = df.corr().iloc[0,1]

# Set the limits of the x and y axis
plt.xlim(10, 80)
plt.ylim(10, 80)

# Extract the independent and dependent variables
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Create and fit the linear regression model
regressor = LinearRegression()
regressor.fit(X, y)

# Make predictions with the model
y_pred = regressor.predict(X)

# Calculate R-squared and RMSE
r_squared = regressor.score(X, y)
rmse = np.sqrt(np.mean((y - y_pred)**2))


# Create the scatter plot with trend line, labels, and title
plt.scatter(X, y, s=50, color='magenta', alpha=0.5)
plt.plot([10, 80], [10, 80], '--', color='black')
plt.plot(X, regressor.predict(X), color='red')  # Use regressor.predict(X) to plot the regression lineplt.xlabel('GNSS-DERIVED PWV(mm)')
plt.xlabel('GNSS-DERIVED PWV(mm)')
plt.ylabel('MODIS-PWV(mm)')


# Set the aspect ratio to make the plot square
plt.gca().set_aspect('equal')

# Create the string with the R-squared, RMSE, bias, and y equation values
textstr = '\n'.join((
     r'$R=%.2f$' % (corr,),
    r'$R^2=%.2f$' % (r_squared,),
    r'$RMSE=%.2f\ mm$' % (rmse,),
    r'$y=%.2fx+%.2f$' % (regressor.coef_[0], regressor.intercept_)))

# Add the text box to the plot
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Calculate the residuals
residuals = y - y_pred

# Determine underestimation or overestimation
if np.mean(residuals) > 0:
    print("The model tends to overestimate the target variable.")
elif np.mean(residuals) < 0:
    print("The model tends to underestimate the target variable.")
else:
    print("The model is unbiased.")
    
# Show the plot
plt.show()


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('ANNUALCORRRAINFALL.csv')

# Extract variables from data
x1 = data['GNSS-DERIVED PWV'].values
x2 = data['MODIS-PWV'].values
y = data['RAINFALL (mm)'].values

# Calculate the correlation coefficients
corr1 = np.corrcoef(x1, y)[0, 1]
corr2 = np.corrcoef(x2, y)[0, 1]

# Set the figure size
plt.figure(figsize=(6, 6))

# Create a scatter plot for x1 vs y
plt.scatter(x1, y, label='GNSS-DERIVED PWV')

# Create a scatter plot for x2 vs y
plt.scatter(x2, y, label='MODIS-PWV')

# Perform linear regression for x1 and y
reg1 = LinearRegression().fit(x1.reshape(-1, 1), y)
y_pred1 = reg1.predict(x1.reshape(-1, 1))
plt.plot(x1, y_pred1, color='blue', linewidth=2)

# Perform linear regression for x2 and y
reg2 = LinearRegression().fit(x2.reshape(-1, 1), y)
y_pred2 = reg2.predict(x2.reshape(-1, 1))
plt.plot(x2, y_pred2, color='orange', linewidth=2)

# Add text annotations for R values
textstr1 = f"R = {corr1:.2f}"
textstr2 = f"R = {corr2:.2f}"
plt.text(0.85, 0.07, textstr1, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round'))
plt.text(0.85, 0.15, textstr2, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round'))

plt.xlabel('PWV (mm)')
plt.ylabel('Rainfall (mm)')
plt.legend(edgecolor='black')

# Set the limits of the x and y axis
plt.xlim(25, 70)
plt.ylim(-5, 35)

plt.show()


# In[2]:


#RAINFALL DRY 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('CORRWRAINFALLDRY.csv')

# Extract variables from data
x1 = data['GNSS-DERIVED PWV'].values
x2 = data['MODIS-PWV'].values
y = data['RAINFALL (mm)'].values

# Calculate the correlation coefficients
corr1 = np.corrcoef(x1, y)[0, 1]
corr2 = np.corrcoef(x2, y)[0, 1]

# Set the figure size
plt.figure(figsize=(6, 6))

# Create a scatter plot for x1 vs y
plt.scatter(x1, y, label='GNSS-DERIVED PWV')

# Create a scatter plot for x2 vs y
plt.scatter(x2, y, label='MODIS-PWV')

# Perform linear regression for x1 and y
reg1 = LinearRegression().fit(x1.reshape(-1, 1), y)
y_pred1 = reg1.predict(x1.reshape(-1, 1))
plt.plot(x1, y_pred1, color='blue', linewidth=2)

# Perform linear regression for x2 and y
reg2 = LinearRegression().fit(x2.reshape(-1, 1), y)
y_pred2 = reg2.predict(x2.reshape(-1, 1))
plt.plot(x2, y_pred2, color='orange', linewidth=2)

# Add text annotations for R values
textstr1 = f"R = {corr1:.2f}"
textstr2 = f"R = {corr2:.2f}"
plt.text(0.85, 0.07, textstr1, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round'))
plt.text(0.85, 0.15, textstr2, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round'))

plt.xlabel('PWV (mm)')
plt.ylabel('Rainfall (mm)')
plt.legend(edgecolor='black')

# Set the limits of the x and y axis
plt.xlim(25, 55)
plt.ylim(-5, 15)

# Set the label for years inside the plot
plt.text(0.33, 0.05, 'DRY SEASON', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))


plt.show()


# In[4]:


#RAINFALL WET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('CORRWRAINFALLWET.csv')

# Extract variables from data
x1 = data['GNSS-DERIVED PWV'].values
x2 = data['MODIS-PWV'].values
y = data['RAINFALL (mm)'].values

# Calculate the correlation coefficients
corr1 = np.corrcoef(x, y1)[0, 1]
corr2 = np.corrcoef(x, y2)[0, 1]

# Set the figure size
plt.figure(figsize=(6, 6))

# Create a scatter plot for x1 vs y
plt.scatter(x1, y, label='GNSS-DERIVED PWV')

# Create a scatter plot for x2 vs y
plt.scatter(x2, y, label='MODIS-PWV')

# Perform linear regression for x1 and y
reg1 = LinearRegression().fit(x1.reshape(-1, 1), y)
y_pred1 = reg1.predict(x1.reshape(-1, 1))
plt.plot(x1, y_pred1, color='blue', linewidth=2)

# Perform linear regression for x2 and y
reg2 = LinearRegression().fit(x2.reshape(-1, 1), y)
y_pred2 = reg2.predict(x2.reshape(-1, 1))
plt.plot(x2, y_pred2, color='orange', linewidth=2)

# Add text annotations for R values
textstr1 = f"R = {corr1:.2f}"
textstr2 = f"R = {corr2:.2f}"
plt.text(0.85, 0.07, textstr1, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round'))
plt.text(0.85, 0.15, textstr2, transform=plt.gca().transAxes, ha='center', va='center', fontsize=14, bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round'))

plt.xlabel('PWV (mm)')
plt.ylabel('Rainfall (mm)')
plt.legend(edgecolor='black')


# Set the label for years inside the plot
plt.text(0.33, 0.05, 'WET SEASON', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# Set the limits of the x and y axis
plt.xlim(35, 70)
plt.ylim(5, 30)
plt.show()


# In[25]:


#CORR OF GNSS-DERIVED PWV WITH ST AND RH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data from CSV file
data = pd.read_csv('ANNUALCORRSTRH.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Extract the independent variables (X) and dependent variables (y1 and y2)
X = data['GNSS-DERIVED PWV'].values
y1 = data['Temp'].values
y2 = data['Rhum'].values

# Calculate R-squared and RMSE for y1
corr1 = np.corrcoef(X, y1)[0, 1]

# Calculate R-squared and RMSE for y2
corr2 = np.corrcoef(X, y2)[0, 1]

# Create a single plot
fig, ax1 = plt.subplots(figsize=(6, 6))

# Configure the plot for y1 (primary)
ax1.scatter(X, y1, color='b')
ax1.set_xlabel('GNSS-DERIVED PWV (mm)')
ax1.set_ylabel('SURFACE TEMPERATURE (°C)', color='b')
ax1.tick_params('y', colors='b')

# Create a secondary y-axis for y2 (secondary variable)
ax2 = ax1.twinx()
ax2.scatter(X, y2, color='r')
ax2.set_ylabel('RELATIVE HUMIDITY (%)', color='r')
ax2.tick_params('y', colors='r')

# Fit a linear regression model for y1
regressor1 = LinearRegression()
regressor1.fit(X.reshape(-1, 1), y1)
y1_pred = regressor1.predict(X.reshape(-1, 1))
ax1.plot(X, y1_pred, color='b')

# Fit a linear regression model for y2
regressor2 = LinearRegression()
regressor2.fit(X.reshape(-1, 1), y2)
y2_pred = regressor2.predict(X.reshape(-1, 1))
ax2.plot(X, y2_pred, color='r')

# Create the text strings for annotations
textstr_1 = f"R = {corr1:.2f}"
textstr_2 = f"R = {corr2:.2f}"

# Adjust vertical positions of text annotations
text_offset = 0.08
ax1.text(0.55, 0.17 - text_offset, textstr_1, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.5))
ax2.text(0.78, 0.17 - text_offset, textstr_2, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.5))

# Set limits for the subplots
ax1.set_xlim(25, 70)
ax1.set_ylim(10, 35)
ax2.set_ylim(50, 100)


# Show the plot
plt.show()


# In[23]:


#CORR OF MODIS-PWV WITH ST AND RH
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data from CSV file
data = pd.read_csv('ANNUALCORRSTRH.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Extract the independent variables (X) and dependent variables (y1 and y2)
X = data['MODIS-PWV'].values
y1 = data['Temp'].values
y2 = data['Rhum'].values

# Calculate R-squared and RMSE for y1
corr1 = np.corrcoef(X, y1)[0, 1]

# Calculate R-squared and RMSE for y2
corr2 = np.corrcoef(X, y2)[0, 1]

# Create a single plot
fig, ax1 = plt.subplots(figsize=(6, 6))

# Configure the plot for y1 (primary)
ax1.scatter(X, y1, color='b')
ax1.set_xlabel('MODIS-PWV (mm)')
ax1.set_ylabel('SURFACE TEMPERATURE (°C)', color='b')
ax1.tick_params('y', colors='b')

# Create a secondary y-axis for y2 (secondary variable)
ax2 = ax1.twinx()
ax2.scatter(X, y2, color='r')
ax2.set_ylabel('RELATIVE HUMIDITY (%)', color='r')
ax2.tick_params('y', colors='r')

# Fit a linear regression model for y1
regressor1 = LinearRegression()
regressor1.fit(X.reshape(-1, 1), y1)
y1_pred = regressor1.predict(X.reshape(-1, 1))
ax1.plot(X, y1_pred, color='b')

# Fit a linear regression model for y2
regressor2 = LinearRegression()
regressor2.fit(X.reshape(-1, 1), y2)
y2_pred = regressor2.predict(X.reshape(-1, 1))
ax2.plot(X, y2_pred, color='r')

# Create the text strings for annotations
textstr_1 = f"R = {corr1:.2f}"
textstr_2 = f"R = {corr2:.2f}"

# Adjust vertical positions of text annotations
text_offset = 0.08
ax1.text(0.55, 0.17 - text_offset, textstr_1, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='blue', alpha=0.5))
ax2.text(0.78, 0.17 - text_offset, textstr_2, transform=ax2.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.5))


# Set limits for the subplots
ax1.set_xlim(20, 70)
ax1.set_ylim(5, 35)
ax2.set_ylim(50, 100)


# Show the plot
plt.show()


# In[ ]:




