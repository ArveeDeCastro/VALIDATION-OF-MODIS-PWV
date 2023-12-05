#!/usr/bin/env python
# coding: utf-8

# In[1]:


#YEAR 2015
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2015.csv')

# convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')


# calculate day of year
df['DATE'] = df['DATE'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(9, 3))
ax.plot(df['DATE'], df['GNSS-DERIVED PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['DATE'], df['MODIS-PWV(mm)'], label='MODIS-PWV', color='c')

# add legend and title
ax.legend(loc='upper left', fontsize='x-small')


# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of the Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))



# show the plot
plt.show()



# In[14]:


#2015
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2015.csv')

# convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')


# calculate day of year
df['DATE'] = df['DATE'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['DATE'], df['GNSS-DERIVED PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['DATE'], df['MODIS-PWV(mm)'], label='MODIS-PWV', color='g')


# add legend and title
ax.legend(loc='upper left', fontsize='x-small', labelspacing=0.05)

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of the Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))

# create a new axis object with same x-axis
ax2 = ax.twinx()

# Replace '-' with 0
df = df.replace('-', '0')

# Convert column to float
df['RAINFALL'] = df['RAINFALL'].astype(float)

# plot a bar graph on the new axis object
ax2. bar(df.index, df['RAINFALL'], label='Rainfall', color='b', alpha=0.5)


# merge the two legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax2.set_ylabel('Rainfall (mm)')

yticks = [0, 20, 40, 60, 80, 100, 120]
ax2.set_yticks(yticks)

y1ticks = [20, 40, 60, 80]
ax.set_yticks(y1ticks)

 # Set the label for years inside the plot
plt.text(0.989, 0.92, '2015', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# show the plot
plt.show()


# In[18]:


#2016
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2016.csv')

# convert date column to datetime type with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# calculate day of year
df['Date'] = df['Date'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['GNSS-derived PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['Date'], df['MODIS-PWV(mm)'], label='MODIS-PWV', color='g')

# add legend and title
ax.legend(loc='upper left', fontsize='x-small')


# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of the Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))

# create a new axis object with same x-axis
ax2 = ax.twinx()

# Replace '-' with 0
df = df.replace('-', '0')

# Convert column to float
df['RAINFALL'] = df['RAINFALL'].astype(float)

# plot a bar graph on the new axis object
ax2.bar(df.index, df['RAINFALL'], label='Rainfall', color='b', alpha=0.5)


# merge the two legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax2.set_ylabel('Rainfall (mm)')

yticks = [0, 20, 40, 60, 80, 100, 120]
ax2.set_yticks(yticks)

y1ticks = [20, 40, 60, 80]
ax.set_yticks(y1ticks)

# Set the label for years inside the plot
plt.text(0.989, 0.92, '2016', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# show the plot
plt.show()


# In[19]:


#2017
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2017.csv')

# convert date column to datetime type with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# calculate day of year
df['Date'] = df['Date'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['Date'], df['GNSS-derived PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['Date'], df['MODIS-PWV'], label='MODIS-PWV', color='g')

# add legend and title
ax.legend(loc='upper left', fontsize='x-small')


# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of the Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))

# create a new axis object with same x-axis
ax2 = ax.twinx()

# Replace '-' with 0
df = df.replace('-', '0')

# Convert column to float
df['RAINFALL'] = df['RAINFALL'].astype(float)

# plot a bar graph on the new axis object
ax2.bar(df.index, df['RAINFALL'], label='Rainfall', color='b', alpha=0.5)


# merge the two legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax2.set_ylabel('Rainfall (mm)')

yticks = [0, 20, 40, 60, 80, 100, 120]
ax2.set_yticks(yticks)

y1ticks = [20, 40, 60, 80]
ax.set_yticks(y1ticks)

# Set the label for years inside the plot
plt.text(0.989, 0.92, '2017', transform=plt.gca().transAxes, fontsize=15, ha='right', va='bottom',
         bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))

# show the plot
plt.show()


# In[35]:


print(df.columns)


# In[102]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Load the data from CSV file
df = pd.read_csv('TIMESERIES2015.csv')


# Define the date parser function
def date_parser(s):
    return datetime.datetime.strptime(s, '%Y-%m-%d')

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(15, 7))
ax.plot(df.index, df['GNSS-DERIVED PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df.index, df['MODIS-PWV(mm)'], label='MODIS-PWV', color='c')

# create a new axis object with same x-axis
ax2 = ax.twinx()

# Replace '-' with 0
df = df.replace('-', '0')

# Convert column to float
df['RAINFALL'] = df['RAINFALL'].astype(float)

# plot a bar graph on the new axis object
ax2.bar(df.index, df['RAINFALL'], label='Rainfall', color='b', alpha=0.5)


# merge the two legends
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax2.set_ylabel('Rainfall (mm)')


yticks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110]
ax2.set_yticks(yticks)

# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))

# show the plot
plt.show()


# In[1]:


#YEAR 2016
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2016.csv')

# convert date column to datetime type with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# calculate day of year
df['Date'] = df['Date'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['GNSS-derived PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['Date'], df['MODIS-PWV(mm)'], label='MODIS-PWV', color='c')

# add legend and title
ax.legend()

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))

# set the x-axis limits and ticks
ax.set_ylim([10, 105])

# show the plot
plt.show()



# In[23]:


#YEAR 2017
import pandas as pd
import matplotlib.pyplot as plt

# load data from CSV file
df = pd.read_csv('TIMESERIES2017.csv')

# convert date column to datetime type with specified format
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')


# calculate day of year
df['Date'] = df['Date'].dt.dayofyear

# create a line plot of two variables
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['Date'], df['GNSS-derived PWV'], label='GNSS-DERIVED PWV', color='#FFA500')
ax.plot(df['Date'], df['MODIS-PWV'], label='MODIS-PWV', color='c')

# add legend and title
ax.legend()

# set labels for y-axes
ax.set_ylabel('PWV (mm)')
ax.set_xlabel('Days of Year')
# set the x-axis limits and ticks
ax.set_xlim([0, 360])
ax.set_xticks(range(0, 361, 30))



# show the plot
plt.show()



# In[34]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'
    
    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
 # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')
    
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year and print maximum and minimum values
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'

    # Calculate and print the maximum and minimum values for each data source
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()
    print(f"Year: {year} - GNSS-DERIVED PWV - Max: {max_gnss}, Min: {min_gnss}")
    print(f"Year: {year} - MODIS-PWV - Max: {max_modis}, Min: {min_modis}")

    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')
    
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Initialize variables for overall max and min values
overall_max_gnss = df['GNSS-DERIVED PWV'].max()
overall_min_gnss = df['GNSS-DERIVED PWV'].min()
overall_max_modis = df['MODIS-PWV(mm)'].max()
overall_min_modis = df['MODIS-PWV(mm)'].min()

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'

    # Calculate maximum and minimum values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()

    # Update overall maximum and minimum values
    overall_max_gnss = max(overall_max_gnss, max_gnss)
    overall_min_gnss = min(overall_min_gnss, min_gnss)
    overall_max_modis = max(overall_max_modis, max_modis)
    overall_min_modis = min(overall_min_modis, min_modis)

    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')
    
# Print the overall maximum and minimum values for both sources
print(f"Overall Max GNSS-DERIVED PWV: {overall_max_gnss}")
print(f"Overall Min GNSS-DERIVED PWV: {overall_min_gnss}")
print(f"Overall Max MODIS-PWV: {overall_max_modis}")
print(f"Overall Min MODIS-PWV: {overall_min_modis}")

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Initialize variables for overall max and min values and their corresponding dates
overall_max_gnss = df['GNSS-DERIVED PWV'].max()
overall_min_gnss = df['GNSS-DERIVED PWV'].min()
overall_max_gnss_date_max = df.loc[df['GNSS-DERIVED PWV'].idxmax()]['DATE']
overall_min_gnss_date_min = df.loc[df['GNSS-DERIVED PWV'].idxmin()]['DATE']

overall_max_modis = df['MODIS-PWV(mm)'].max()
overall_min_modis = df['MODIS-PWV(mm)'].min()
overall_max_modis_date_max = df.loc[df['MODIS-PWV(mm)'].idxmax()]['DATE']
overall_min_modis_date_min = df.loc[df['MODIS-PWV(mm)'].idxmin()]['DATE']

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'

    # Calculate maximum and minimum values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()

    # Update overall maximum and minimum values if necessary
    if max_gnss > overall_max_gnss:
        overall_max_gnss = max_gnss
        overall_max_gnss_date_max = df_year.loc[df_year['GNSS-DERIVED PWV'].idxmax()]['DATE']
    if min_gnss < overall_min_gnss:
        overall_min_gnss = min_gnss
        overall_min_gnss_date_min = df_year.loc[df_year['GNSS-DERIVED PWV'].idxmin()]['DATE']
    if max_modis > overall_max_modis:
        overall_max_modis = max_modis
        overall_max_modis_date_max = df_year.loc[df_year['MODIS-PWV(mm)'].idxmax()]['DATE']
    if min_modis < overall_min_modis:
        overall_min_modis = min_modis
        overall_min_modis_date_min = df_year.loc[df_year['MODIS-PWV(mm)'].idxmin()]['DATE']

    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')

# Print the overall maximum and minimum values for both sources and their corresponding dates
print(f"Overall Max GNSS-DERIVED PWV: {overall_max_gnss} (Date: {overall_max_gnss_date_max})")
print(f"Overall Min GNSS-DERIVED PWV: {overall_min_gnss} (Date: {overall_min_gnss_date_min})")
print(f"Overall Max MODIS-PWV: {overall_max_modis} (Date: {overall_max_modis_date_max})")
print(f"Overall Min MODIS-PWV: {overall_min_modis} (Date: {overall_min_modis_date_min})")

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Initialize variables for overall max and min values
overall_max_gnss = df['GNSS-DERIVED PWV'].max()
overall_min_gnss = df['GNSS-DERIVED PWV'].min()
overall_max_modis = df['MODIS-PWV(mm)'].max()
overall_min_modis = df['MODIS-PWV(mm)'].min()

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'

    # Calculate maximum and minimum values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()

    # Calculate variation from initial max and min values
    variation_max_gnss = max_gnss - overall_max_gnss
    variation_min_gnss = min_gnss - overall_min_gnss
    variation_max_modis = max_modis - overall_max_modis
    variation_min_modis = min_modis - overall_min_modis

    # Print the variation in maximum and minimum values
    print(f"Year: {year} - Variation in Max GNSS-DERIVED PWV: {variation_max_gnss}, Min GNSS-DERIVED PWV: {variation_min_gnss}")
    print(f"Year: {year} - Variation in Max MODIS-PWV: {variation_max_modis}, Min MODIS-PWV: {variation_min_modis}")

    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Initialize variables for overall max and min values
overall_max_gnss = df['GNSS-DERIVED PWV'].max()
overall_min_gnss = df['GNSS-DERIVED PWV'].min()
overall_max_modis = df['MODIS-PWV(mm)'].max()
overall_min_modis = df['MODIS-PWV(mm)'].min()

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0])  # Use the first color for 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1])  # Use the second color for 'MODIS-PWV'

    # Calculate maximum and minimum values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()

    # Calculate variation from overall max and min values
    variation_max_gnss = max_gnss - overall_max_gnss
    variation_min_gnss = min_gnss - overall_min_gnss
    variation_max_modis = max_modis - overall_max_modis
    variation_min_modis = min_modis - overall_min_modis

    # Print the variation in maximum and minimum values
    print(f"Year: {year} - Variation in Max GNSS-DERIVED PWV: {variation_max_gnss}, Min GNSS-DERIVED PWV: {variation_min_gnss}")
    print(f"Year: {year} - Variation in Max MODIS-PWV: {variation_max_modis}, Min MODIS-PWV: {variation_min_modis}")

    # Add legend for the first subplot only
    if i == 0:
        ax.plot([], [], label='GNSS-DERIVED PWV', color=line_colors[0])  # Empty plot for 'GNSS-DERIVED PWV' legend
        ax.plot([], [], label='MODIS-PWV', color=line_colors[1])  # Empty plot for 'MODIS-PWV' legend
        ax.legend(loc='upper left', fontsize='xx-small')
        
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Create a line plot for each year
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 5), sharex=False)

# Iterate over each year and create a subplot
years = [2015, 2016, 2017]
line_colors = ['#FFA500', 'c']  # Different colors for 'GNSS-DERIVED PWV' and 'MODIS-PWV'

for i, year in enumerate(years):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex
    ax = axes[i]
    
    ax.plot(df_year['DATE'], df_year['GNSS-DERIVED PWV'], color=line_colors[0], label='GNSS-DERIVED PWV')  # Plot 'GNSS-DERIVED PWV'
    ax.plot(df_year['DATE'], df_year['MODIS-PWV(mm)'], color=line_colors[1], label='MODIS-PWV')  # Plot 'MODIS-PWV'

    # Calculate maximum and minimum values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()

    # Find dates for maximum and minimum values
    date_max_gnss = df_year.loc[df_year['GNSS-DERIVED PWV'].idxmax()]['DATE']
    date_min_gnss = df_year.loc[df_year['GNSS-DERIVED PWV'].idxmin()]['DATE']
    date_max_modis = df_year.loc[df_year['MODIS-PWV(mm)'].idxmax()]['DATE']
    date_min_modis = df_year.loc[df_year['MODIS-PWV(mm)'].idxmin()]['DATE']

    # Print actual maximum and minimum values for each data source and their corresponding dates
    print(f"Year: {year} - Max GNSS-DERIVED PWV: {max_gnss} (Date: {date_max_gnss})")
    print(f"Year: {year} - Min GNSS-DERIVED PWV: {min_gnss} (Date: {date_min_gnss})")
    print(f"Year: {year} - Max MODIS-PWV: {max_modis} (Date: {date_max_modis})")
    print(f"Year: {year} - Min MODIS-PWV: {min_modis} (Date: {date_min_modis})")

    # Add legend for each subplot
    ax.legend(loc='upper left', fontsize='xx-small')
    
    # Set the label for years inside the plot
    ax.text(0.99, 0.80, str(year), transform=ax.transAxes, fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='k', boxstyle='round,pad=0.2'))
    
    # Set labels for y-axes for the last subplot only
    if i == 2:
        ax.set_ylabel('PWV (mm)')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.2)

# Show the plot
plt.show()


# In[7]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize a dictionary to store min and max values for each year
yearly_values = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Calculate minimum and maximum values for each data source for the current year
    min_gnss = df_year['GNSS-DERIVED PWV'].min()
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    min_modis = df_year['MODIS-PWV(mm)'].min()
    max_modis = df_year['MODIS-PWV(mm)'].max()

    # Store the min and max values in the dictionary
    yearly_values[year] = {
        'GNSS-DERIVED PWV': {'min': min_gnss, 'max': max_gnss},
        'MODIS-PWV': {'min': min_modis, 'max': max_modis}
    }

# Display the min and max values for each year
for year, values in yearly_values.items():
    print(f"Year: {year}")
    print(f"Min GNSS-DERIVED PWV: {values['GNSS-DERIVED PWV']['min']}, Max GNSS-DERIVED PWV: {values['GNSS-DERIVED PWV']['max']}")
    print(f"Min MODIS-PWV: {values['MODIS-PWV']['min']}, Max MODIS-PWV: {values['MODIS-PWV']['max']}")


# In[8]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize a dictionary to store min and max peak values for each year
yearly_values = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Find the indices of the peak minimum and maximum values for each data source for the current year
    idx_min_gnss = df_year['GNSS-DERIVED PWV'].idxmin()
    idx_max_gnss = df_year['GNSS-DERIVED PWV'].idxmax()
    idx_min_modis = df_year['MODIS-PWV(mm)'].idxmin()
    idx_max_modis = df_year['MODIS-PWV(mm)'].idxmax()

    # Get the corresponding peak minimum and maximum values along with their dates for each data source
    min_gnss = df_year.loc[idx_min_gnss, 'GNSS-DERIVED PWV']
    max_gnss = df_year.loc[idx_max_gnss, 'GNSS-DERIVED PWV']
    min_modis = df_year.loc[idx_min_modis, 'MODIS-PWV(mm)']
    max_modis = df_year.loc[idx_max_modis, 'MODIS-PWV(mm)']

    # Store the peak values in the dictionary
    yearly_values[year] = {
        'GNSS-DERIVED PWV': {'min': min_gnss, 'max': max_gnss},
        'MODIS-PWV': {'min': min_modis, 'max': max_modis}
    }

# Display the peak minimum and maximum values for each year
for year, values in yearly_values.items():
    print(f"Year: {year}")
    print(f"Min Peak GNSS-DERIVED PWV: {values['GNSS-DERIVED PWV']['min']}, Max Peak GNSS-DERIVED PWV: {values['GNSS-DERIVED PWV']['max']}")
    print(f"Min Peak MODIS-PWV: {values['MODIS-PWV']['min']}, Max Peak MODIS-PWV: {values['MODIS-PWV']['max']}")


# In[9]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize a dictionary to store max peak values for each year
yearly_max_values = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Find the indices of the peak maximum values for each data source for the current year
    idx_max_gnss = df_year['GNSS-DERIVED PWV'].idxmax()
    idx_max_modis = df_year['MODIS-PWV(mm)'].idxmax()

    # Get the corresponding peak maximum values for each data source
    max_gnss = df_year.loc[idx_max_gnss, 'GNSS-DERIVED PWV']
    max_modis = df_year.loc[idx_max_modis, 'MODIS-PWV(mm)']

    # Store the peak maximum values in the dictionary
    yearly_max_values[year] = {
        'GNSS-DERIVED PWV': max_gnss,
        'MODIS-PWV': max_modis
    }

# Find the minimum and maximum of the maximum peak values across all years
min_max_peak_gnss = min(yearly_max_values.values(), key=lambda x: x['GNSS-DERIVED PWV'])
max_max_peak_gnss = max(yearly_max_values.values(), key=lambda x: x['GNSS-DERIVED PWV'])
min_max_peak_modis = min(yearly_max_values.values(), key=lambda x: x['MODIS-PWV'])
max_max_peak_modis = max(yearly_max_values.values(), key=lambda x: x['MODIS-PWV'])

# Display the minimum and maximum of the maximum peak values across years
print("Minimum of maximum peak GNSS-DERIVED PWV:", min_max_peak_gnss['GNSS-DERIVED PWV'])
print("Maximum of maximum peak GNSS-DERIVED PWV:", max_max_peak_gnss['GNSS-DERIVED PWV'])
print("Minimum of maximum peak MODIS-PWV:", min_max_peak_modis['MODIS-PWV'])
print("Maximum of maximum peak MODIS-PWV:", max_max_peak_modis['MODIS-PWV'])


# In[10]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize a dictionary to store max peak values for each year
yearly_max_values = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Find the maximum peak values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    max_modis = df_year['MODIS-PWV(mm)'].max()

    # Store the maximum peak values in the dictionary
    yearly_max_values[year] = {
        'GNSS-DERIVED PWV': max_gnss,
        'MODIS-PWV': max_modis
    }

# Display the maximum peak values for each year
for year, values in yearly_max_values.items():
    print(f"Year: {year}")
    print(f"Maximum Peak GNSS-DERIVED PWV: {values['GNSS-DERIVED PWV']}")
    print(f"Maximum Peak MODIS-PWV: {values['MODIS-PWV']}")

# Find the minimum and maximum of the maximum peak values across years
min_max_peak_gnss = min(yearly_max_values.values(), key=lambda x: x['GNSS-DERIVED PWV'])
max_max_peak_gnss = max(yearly_max_values.values(), key=lambda x: x['GNSS-DERIVED PWV'])
min_max_peak_modis = min(yearly_max_values.values(), key=lambda x: x['MODIS-PWV'])
max_max_peak_modis = max(yearly_max_values.values(), key=lambda x: x['MODIS-PWV'])

# Display the minimum and maximum of the maximum peak values across years
print("\nMinimum of maximum peak GNSS-DERIVED PWV:", min_max_peak_gnss['GNSS-DERIVED PWV'])
print("Maximum of maximum peak GNSS-DERIVED PWV:", max_max_peak_gnss['GNSS-DERIVED PWV'])
print("Minimum of maximum peak MODIS-PWV:", min_max_peak_modis['MODIS-PWV'])
print("Maximum of maximum peak MODIS-PWV:", max_max_peak_modis['MODIS-PWV'])


# In[11]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize dictionaries to store max peak values for each year
yearly_max_values_gnss = {}
yearly_max_values_modis = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Find the maximum peak values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    max_modis = df_year['MODIS-PWV(mm)'].max()

    # Store the maximum peak values for each year in separate dictionaries
    yearly_max_values_gnss[year] = max_gnss
    yearly_max_values_modis[year] = max_modis

# Find the minimum and maximum of the maximum peak values across years for GNSS-DERIVED PWV and MODIS-PWV
min_max_peak_gnss = min(yearly_max_values_gnss.values())
max_max_peak_gnss = max(yearly_max_values_gnss.values())
min_max_peak_modis = min(yearly_max_values_modis.values())
max_max_peak_modis = max(yearly_max_values_modis.values())

# Display the minimum and maximum of the maximum peak values across years for each data source
print("Minimum of maximum peak GNSS-DERIVED PWV:", min_max_peak_gnss)
print("Maximum of maximum peak GNSS-DERIVED PWV:", max_max_peak_gnss)
print("Minimum of maximum peak MODIS-PWV:", min_max_peak_modis)
print("Maximum of maximum peak MODIS-PWV:", max_max_peak_modis)


# In[12]:


import pandas as pd

# Load data from CSV file
df = pd.read_csv('TIMESERIES15-17.csv')

# Convert date column to datetime type with specified format
df['DATE'] = pd.to_datetime(df['DATE'], format='%d/%m/%Y')

# Initialize dictionaries to store max peak values for each year
yearly_max_values_gnss = {}
yearly_max_values_modis = {}

# Iterate over each year
for year in range(2015, 2018):
    df_year = df[pd.DatetimeIndex(df['DATE']).year == year]  # Filter data for each year using pd.DatetimeIndex

    # Find the maximum peak values for each data source for the current year
    max_gnss = df_year['GNSS-DERIVED PWV'].max()
    max_modis = df_year['MODIS-PWV(mm)'].max()

    # Store the maximum peak values for each year in separate dictionaries
    yearly_max_values_gnss[year] = max_gnss
    yearly_max_values_modis[year] = max_modis

# Display the minimum and maximum of the maximum peak values for each year for GNSS-DERIVED PWV
print("Maximum Peak GNSS-DERIVED PWV Values:")
for year, value in yearly_max_values_gnss.items():
    print(f"Year {year}: {value}")

# Find the minimum and maximum of the maximum peak values for GNSS-DERIVED PWV across years
min_max_peak_gnss = min(yearly_max_values_gnss.values())
max_max_peak_gnss = max(yearly_max_values_gnss.values())

# Display the minimum and maximum of the maximum peak values for GNSS-DERIVED PWV
print("\nMinimum of maximum peak GNSS-DERIVED PWV:", min_max_peak_gnss)
print("Maximum of maximum peak GNSS-DERIVED PWV:", max_max_peak_gnss)

# Display the minimum and maximum of the maximum peak values for each year for MODIS-PWV
print("\nMaximum Peak MODIS-PWV Values:")
for year, value in yearly_max_values_modis.items():
    print(f"Year {year}: {value}")

# Find the minimum and maximum of the maximum peak values for MODIS-PWV across years
min_max_peak_modis = min(yearly_max_values_modis.values())
max_max_peak_modis = max(yearly_max_values_modis.values())

# Display the minimum and maximum of the maximum peak values for MODIS-PWV
print("\nMinimum of maximum peak MODIS-PWV:", min_max_peak_modis)
print("Maximum of maximum peak MODIS-PWV:", max_max_peak_modis)


# In[ ]:




