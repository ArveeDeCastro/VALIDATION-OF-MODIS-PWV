#!/usr/bin/env python
# coding: utf-8

# In[1]:


#ANNUAL CORRELATION WITH TWO VARIABLES W/ LINEAR REGRESSION
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('CORR2016.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Select the two variables to calculate correlation and create a new dataframe
X = data['GNSS-derived PWV']
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
plt.plot([0, 80], [0, 80], '--', color='black')
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


# In[1]:


#DRY SEASON 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('DRYSEASON2016.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Select the two variables to calculate correlation and create a new dataframe
x = data['GNSS-derived PWV']
y = data['MODIS-PWV']
df = pd.concat([x, y], axis=1)

# Calculate the correlation coefficient
corr = df.corr().iloc[0,1]

# Fit a polynomial regression line to the data
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Create the scatter plot with trend line, labels, and title
plt.scatter(x, y, s=50, color='magenta', alpha=0.5)
plt.plot([10, 80], [10, 80], '--', color='black')
plt.plot(x, p(x), color='red')
plt.xlabel('GNSS-DERIVED PWV(mm)')
plt.ylabel('MODIS-PWV(mm)')

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
rmse = np.sqrt(np.mean((y  - y_pred)**2))

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

# Set the aspect ratio to make the plot square
plt.gca().set_aspect('equal')

# Calculate the residuals
residuals = y - y_pred

# Determine underestimation or overestimation
if np.mean(residuals) > 0:
    print("The model tends to overestimate the target variable.")
elif np.mean(residuals) < 0:
    print("The model tends to underestimate the target variable.")
else:
    print("The model is unbiased.")

    # Set the label for years inside the plot
plt.text(0.96, 0.05, 'DRY SEASON', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# Show the plot
plt.show()


# In[3]:


#WET SEASON 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file
data = pd.read_csv('WETSEASON2016.csv')

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Select the two variables to calculate correlation and create a new dataframe
x = data['GNSS-derived PWV']
y = data['MODIS-PWV']
df = pd.concat([x, y], axis=1)

# Calculate the correlation coefficient
corr = df.corr().iloc[0,1]

# Fit a polynomial regression line to the data
z = np.polyfit(x, y, 1)
p = np.poly1d(z)

# Create the scatter plot with trend line, labels, and title
plt.scatter(x, y, s=50, color='magenta', alpha=0.5)
plt.plot([0, 80], [0, 80], '--', color='black')
plt.plot(x, p(x), color='red')
plt.xlabel('GNSS-DERIVED PWV(mm)')
plt.ylabel('MODIS-PWV(mm)')

# Set the limits of the x and y axis
plt.xlim(10, 80)
plt.ylim(10, 80)

# Set the aspect ratio to make the plot square
plt.gca().set_aspect('equal')

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
    
    # Set the label for years inside the plot
plt.text(0.96, 0.05, 'WET SEASON', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# Show the plot
plt.show()


# In[ ]:


#CORR OF PWV WITH RAINFALL
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load the data from CSV file
data = pd.read_csv('CORRWITHRAINFALL2015.csv')

# Load data from CSV file
data = pd.read_csv('CORRWITHRAINFALL2015.csv')

# Extract X and y values from data
x = data['RAINFALL (mm)'].values
y1 = data['GNSS-DERIVED PWV'].values
y2 = data['MODIS-PWV'].values

# Create a single plot
fig, ax = plt.subplots(figsize=(6, 6))

# Perform linear regression and plot data for y1
reg1 = LinearRegression().fit(x.reshape(-1, 1), y1)
y1_pred = reg1.predict(x.reshape(-1, 1))

# Calculate R-squared and RMSE for y1
corr = np.corrcoef(x, y1)[0, 1]
r_squared_1 = reg1.score(x.reshape(-1, 1), y1)
rmse_1 = np.sqrt(mean_squared_error(y1, y1_pred))
bias_1 = np.mean(y1 - y1_pred)

# Perform linear regression and plot data for y2
reg2 = LinearRegression().fit(x.reshape(-1, 1), y2)
y2_pred = reg2.predict(x.reshape(-1, 1))

# Calculate R-squared and RMSE for y2
corr = np.corrcoef(x, y2)[0, 1]
r_squared_2 = reg2.score(x.reshape(-1, 1), y2)
rmse_2 = np.sqrt(mean_squared_error(y2, y2_pred))
bias_2 = np.mean(y2 - y2_pred)

# Plot data and regression line for y1
ax.scatter(x, y1, color='c', label='GNSS-DERIVED PWV')
ax.plot(x, y1_pred, color='cyan')

# Plot data and regression line for y2
ax.scatter(x, y2, color='m', label='MODIS-PWV')
ax.plot(x, y2_pred, color='magenta')

# Create the text strings for annotations
textstr_1 = f"R: {corr:.2f}"
textstr_2 = f"R: {corr:.2f}"

# Adjust vertical positions of text annotations
text_offset = 0.08
ax.text(0.50, 0.35 - text_offset, textstr_1, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='cyan', alpha=0.5))
ax.text(0.69, 0.35 - text_offset, textstr_2, transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', edgecolor='magenta', alpha=0.5))

# Create the custom legend text
custom_legend = 'R² = 0.98\n' \
               'RMSE = 1.41 mm\n' \
               'Bias = -10.98\n' \
               'y = 1.14x1) + -0.17x2 + -10.98'

# Add the text box to the plot
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
plt.text(0.50, 0.19, custom_legend, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props, fontdict={'style': 'italic'})

ax.set_xlabel('Rainfall (mm)')
ax.set_ylabel('PWV (mm)')

# Set the limits of the x and y axis
ax.set_xlim(0, 30)
ax.set_ylim(0, 80)
ax.plot([0, 30], [0, 80], '--', color='black')


# Add legend
ax.legend()


# Show the plot
plt.show()



# In[6]:


# Step 2: Import required libraries
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# Step 3: Read the CSV file
# Replace 'data.csv' with the actual path to your CSV file if it's located in a different directory.
df = pd.read_csv('CORR2016.csv')

# Assuming you have columns 'y_pred' and 'y_obs' in your DataFrame
y_pred = df['GNSS-derived PWV']
y_obs = df['MODIS-PWV(mm)']

# Step 4: Calculate the residuals
residuals = y_obs - y_pred

# Step 5: Create the Q-Q plot
fig, ax = plt.subplots(figsize=(6, 6))
sm.qqplot(residuals, line='s', ax=ax)
ax.set_title('Q-Q Plot of Residuals')
plt.show()


# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import calendar

# Load the data from CSV file, specifying the date format
data = pd.read_csv('CORR2016WDATE.csv', parse_dates=['Date'], dayfirst=True)

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Specify the time frame you want to filter (replace start_date and end_date with your desired dates)
start_date = '2016-01-01'
end_date = '2016-12-31'

# Create a grid of subplots (4 rows, 3 columns)
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 10))
months = data.resample('M')

# Iterate through each month (1 to 12)
for i, (month_start, month_data) in enumerate(months):
    # Check if there is data for the current month
    if not month_data.empty:
        # Filter the data for the specific time frame
        data = month_data[start_date:end_date]
        
        # Check for missing values in your data
        print(data.isnull().sum())
        
        # Remove any rows containing missing values
        data = data.dropna()
        
        # Select the two variables to calculate correlation and create a new dataframe
        X = data['GNSS-derived PWV']
        y = data['MODIS-PWV(mm)']
        
        # Calculate the correlation coefficient
        corr = X.corr(y)
        
        # Create and fit the linear regression model
        regressor = LinearRegression()
        regressor.fit(X.values.reshape(-1, 1), y)  # Reshape X to a 2D array for LinearRegression
        
        # Make predictions with the model
        y_pred = regressor.predict(X.values.reshape(-1, 1))
        
        # Create a new subplot for the current month
        ax = axes[i // 3, i % 3]
        ax.scatter(X, y, s=50, color='magenta', alpha=0.5)
        ax.plot([0, 80], [0, 80], '--', color='black')
        ax.plot(X, y_pred, color='red')  # Use y_pred to plot the regression line
        ax.set_xlim(10, 80)
        ax.set_ylim(10, 80)
        ax.set_title(calendar.month_name[month_start.month])  # Use month name as title
        ax.set_aspect('equal')
        
        # Calculate R-squared and RMSE
        r_squared = regressor.score(X.values.reshape(-1, 1), y)
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        # Create the string with the R-squared, RMSE, bias, and y equation values
        textstr = '\n'.join((
             r'$R=%.2f$' % (corr,),
            r'$R^2=%.2f$' % (r_squared,),
            r'$RMSE=%.2f\ mm$' % (rmse,),
            r'$y=%.2fx+%.2f$' % (regressor.coef_[0], regressor.intercept_)))
        
        # Add the text box to the plot
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=6,
                verticalalignment='top', bbox=props)
        
        # Set the aspect ratio to make the plot square
        ax.set_aspect('equal')

# Adjust layout to prevent overlap and make plots closer
plt.tight_layout(pad=1.0)

# Set common x and y labels
fig.text(0.51, 0.0009, 'GNSS-DERIVED PWV(mm)', ha='center', fontsize=12)
fig.text(0.001, 0.5, 'MODIS-PWV(mm)', va='center', rotation='vertical', fontsize=12)

# Show the plots
plt.show()


# In[ ]:





# In[2]:


#MONTHLY COMPARISONS 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data from CSV file, specifying the date format
data = pd.read_csv('CORR2016WDATE.csv', parse_dates=['Date'], dayfirst=True)

# Set the 'Date' column as the index
data.set_index('Date', inplace=True)

# Specify the time frame you want to filter (replace start_date and end_date with your desired dates)
start_date = '2016-01-01'
end_date = '2016-01-31'

# Filter the data for the specific time frame
data = data[start_date:end_date]

# Check for missing values in your data
print(data.isnull().sum())

# Remove any rows containing missing values
data = data.dropna()

# Select the two variables to calculate correlation and create a new dataframe
X = data['GNSS-derived PWV']
y = data['MODIS-PWV(mm)']

# Calculate the correlation coefficient
corr = X.corr(y)

# Set the limits of the x and y axis
plt.xlim(10, 80)
plt.ylim(10, 80)

# Create and fit the linear regression model
regressor = LinearRegression()
regressor.fit(X.values.reshape(-1, 1), y)  # Reshape X to a 2D array for LinearRegression

# Make predictions with the model
y_pred = regressor.predict(X.values.reshape(-1, 1))

# Calculate R-squared and RMSE
r_squared = regressor.score(X.values.reshape(-1, 1), y)
rmse = np.sqrt(np.mean((y - y_pred)**2))

# Create the scatter plot with trend line, labels, and title
plt.scatter(X, y, s=50, color='blue', alpha=0.5)
plt.plot([0, 80], [0, 80], '--', color='black')
plt.plot(X, y_pred, color='red')  # Use y_pred to plot the regression line
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

