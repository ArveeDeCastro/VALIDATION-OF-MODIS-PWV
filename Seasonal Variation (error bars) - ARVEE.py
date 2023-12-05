#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('MONTHLYMODIS2015CSV.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars
fig, ax = plt.subplots()
ax.errorbar(df.index, df['MODIS-PWV'], yerr=df['STD'], fmt='-o', color='m', ecolor='m', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('MODIS-PWV')

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('MONTHLYGNSSPWV2015CSV.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars
fig, ax = plt.subplots()
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD'], fmt='-o', color='c', ecolor='c', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('GNSS-DERIVED PWV')

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

plt.show()


# In[ ]:





# In[19]:


#2015
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATION (ERROR BARS)2015.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots()
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='c', ecolor='c', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='m', ecolor='m', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend(loc='upper left', fontsize='small')

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Set the label for years inside the plot
ax.text(0.96, 0.90, str(2015), transform=ax.transAxes, fontsize=15, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas dataframe
df = pd.read_csv('data.csv')

# Group the data by the x_variable
grouped = df.groupby('x_variable')

# Calculate the mean and standard deviation for each group
means = grouped[['y_variable_1', 'y_variable_2', 'y_variable_3']].mean()
stds = grouped[['y_variable_1', 'y_variable_2', 'y_variable_3']].std()

# Plot the line graph with error bars representing the standard deviation
plt.errorbar(means.index, means['y_variable_1'], yerr=stds['y_variable_1'], fmt='-o', label='y_variable_1')
plt.errorbar(means.index, means['y_variable_2'], yerr=stds['y_variable_2'], fmt='-o', label='y_variable_2')
plt.errorbar(means.index, means['y_variable_3'], yerr=stds['y_variable_3'], fmt='-o', label='y_variable_3')
plt.xlabel('x_variable')
plt.ylabel('y_variable')
plt.legend()
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATION (ERROR BARS)2015.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax1 = plt.subplots()
ax1.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='c', ecolor='c', elinewidth=1, capsize=2)
ax1.errorbar(df.index, df['MODIS-PWV'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='m', ecolor='m', elinewidth=1, capsize=2)

# Create a second y-axis
ax2 = ax1.twinx()

# Plot the third variable as a line graph on the second y-axis
ax2.plot(df.index, df['RAINFALL (mm)'], color='b', label='Rainfall')

# Set the x-axis label to "Month"
ax1.set_xlabel('Months')

# Set the y-axis label to "PWV (mm)"
ax1.set_ylabel('PWV (mm)')

# Set the y-axis label to "Third Variable"
ax2.set_ylabel('Rainfall (mm)')

# merge the two legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', fontsize=9)


# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # month abbreviation
ax1.xaxis.set_major_locator(months)
ax1.xaxis.set_major_formatter(months_fmt)

plt.show()


# In[17]:


#2016
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATIONS 2016.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots()
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='c', ecolor='c', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV(mm)'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='m', ecolor='m', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend(loc='upper left', fontsize='small')

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Set the label for years inside the plot
ax.text(0.96, 0.90, str(2016), transform=ax.transAxes, fontsize=15, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

plt.show()


# In[18]:


#2017
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATIONS 2017.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots()
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='c', ecolor='c', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV(mm)'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='m', ecolor='m', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend(loc='upper left', fontsize='small')

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Set the label for years inside the plot
ax.text(0.96, 0.90, str(2017), transform=ax.transAxes, fontsize=15, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

plt.show()


# In[18]:


#2015
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATION (ERROR BARS)2015.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots(figsize=(12, 3)) 
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='#FFA500', ecolor='#FFA500', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='g', ecolor='g', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend()

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Calculate the variability for each month
df['Variability'] = df['GNSS-DERIVED PWV'] - df['MODIS-PWV']

# Print the variability values for each month
for month, variability in zip(df.index, df['Variability']):
    print(f"Month: {month.strftime('%b')}, Variability: {variability:.2f} mm")
    
# Set the label for years inside the plot
plt.text(0.10, 0.80, '2015', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))


plt.show()


# In[24]:


#2016
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATIONS 2016.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots(figsize=(12, 3)) 
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='#FFA500', ecolor='#FFA500', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV(mm)'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='g', ecolor='g', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend()

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Calculate the variability for each month
df['Variability'] = df['GNSS-DERIVED PWV'] - df['MODIS-PWV(mm)']

# Print the variability values for each month
for month, variability in zip(df.index, df['Variability']):
    print(f"Month: {month.strftime('%b')}, Variability: {variability:.2f} mm")

# Set the label for years inside the plot
plt.text(0.96, 0.10, '2016', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))


plt.show()


# In[25]:


#2017
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
os.getcwd()


# Load the data from the CSV file into a pandas DataFrame
df = pd.read_csv('SEASONAL VARIATIONS 2017.csv')

# Convert the date column to a pandas datetime object and set it as the index
df['MONTH'] = pd.to_datetime(df['MONTH'])
df.set_index('MONTH', inplace=True)

# Create the line graph with standard deviation as error bars for two variables
fig, ax = plt.subplots(figsize=(12, 3)) 
ax.errorbar(df.index, df['GNSS-DERIVED PWV'], yerr=df['STD_A'], fmt='-o', label='GNSS-DERIVED PWV', color='#FFA500', ecolor='#FFA500', elinewidth=1, capsize=2)
ax.errorbar(df.index, df['MODIS-PWV(mm)'], yerr=df['STD_B'], fmt='-o', label='MODIS-PWV', color='g', ecolor='g', elinewidth=1, capsize=2)

# Set the x-axis label to "Month"
ax.set_xlabel('Months')

# Set the y-axis label to "y_variable"
ax.set_ylabel('PWV (mm)')

# Add a legend
ax.legend()

# Format the y-axis tick labels to display only the month name
months = mdates.MonthLocator()  # every month
months_fmt = mdates.DateFormatter('%b')  # full month name
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(months_fmt)

# Calculate the variability for each month
df['Variability'] = df['GNSS-DERIVED PWV'] - df['MODIS-PWV(mm)']

# Print the variability values for each month
for month, variability in zip(df.index, df['Variability']):
    print(f"Month: {month.strftime('%b')}, Variability: {variability:.2f} mm")

# Set the label for years inside the plot
plt.text(0.10, 0.80, '2017', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))


plt.show()


# In[1]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Calculate mean values for each data source over the three years
mean_gnss = df['GNSS-DERIVED PWV'].mean()
mean_modis = df['MODIS-PWV(mm)'].mean()

# Display the mean values
print("Mean GNSS-DERIVED PWV:", mean_gnss)
print("Mean MODIS-PWV:", mean_modis)

# Calculate the mean difference between the two data sources
mean_difference = mean_gnss - mean_modis

# Display the mean difference
print("Mean Difference between GNSS-DERIVED PWV and MODIS-PWV:", mean_difference)


# In[2]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Calculate daily mean values for each data source over the three years
daily_mean_gnss = df.groupby('DATE')['GNSS-DERIVED PWV'].mean()
daily_mean_modis = df.groupby('DATE')['MODIS-PWV(mm)'].mean()

# Calculate overall mean values
mean_gnss = daily_mean_gnss.mean()
mean_modis = daily_mean_modis.mean()

# Display the mean values
print("Mean GNSS-DERIVED PWV (daily):", mean_gnss)
print("Mean MODIS-PWV (daily):", mean_modis)

# Calculate the mean difference between the two data sources
mean_difference = mean_gnss - mean_modis

# Display the mean difference
print("Mean Difference between GNSS-DERIVED PWV and MODIS-PWV (daily):", mean_difference)


# In[1]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Extract year and month from the date
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month

# Calculate monthly mean values for each data source and year
monthly_mean_gnss = df.groupby(['Year', 'Month'])['GNSS-DERIVED PWV'].mean()
monthly_mean_modis = df.groupby(['Year', 'Month'])['MODIS-PWV(mm)'].mean()

# Calculate the range of monthly mean values for each data source and year
range_gnss = monthly_mean_gnss.groupby('Year').max() - monthly_mean_gnss.groupby('Year').min()
range_modis = monthly_mean_modis.groupby('Year').max() - monthly_mean_modis.groupby('Year').min()

# Display the range of monthly mean values for each year
print("Range of Monthly Mean GNSS-DERIVED PWV:")
print(range_gnss)
print("\nRange of Monthly Mean MODIS-PWV:")
print(range_modis)


# In[3]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Extract year and month from the date
df['Year'] = df['DATE'].dt.year
df['Month'] = df['DATE'].dt.month

# Calculate monthly mean values for each data source and year
monthly_mean_gnss = df.groupby(['Year', 'Month'])['GNSS-DERIVED PWV'].mean()
monthly_mean_modis = df.groupby(['Year', 'Month'])['MODIS-PWV(mm)'].mean()

# Calculate the range, minimum, and maximum of monthly mean values for each data source and year
range_gnss = monthly_mean_gnss.groupby('Year').agg(['min', 'max', lambda x: x.max() - x.min()])
range_modis = monthly_mean_modis.groupby('Year').agg(['min', 'max', lambda x: x.max() - x.min()])

# Display the range, minimum, and maximum of monthly mean values for each year
print("Statistics for Monthly Mean GNSS-DERIVED PWV:")
print(range_gnss)
print("\nStatistics for Monthly Mean MODIS-PWV:")
print(range_modis)


# In[ ]:




