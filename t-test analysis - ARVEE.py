#!/usr/bin/env python
# coding: utf-8

# In[40]:


#2015-2017
import pandas as pd
from scipy.stats import ttest_ind

# Load data from Excel file
# Replace 'ANOVA FOR 2015.xlsx' with the actual name of your Excel file
df = pd.read_excel('T-TEST FOR 2015-17.xlsx')

# Check for missing values and handle them if needed
if df.isnull().sum().any():
    # If there are missing values, you can choose to drop or fill them
    df = df.dropna()  # Use df.fillna() if you want to fill missing values

# Ensure numeric data types for 'GNSS-DERIVED PWV' and 'MODIS-PWV(mm)'
df['GNSS-DERIVED PWV'] = pd.to_numeric(df['GNSS-DERIVED PWV'], errors='coerce')
df['MODIS-PWV(mm)'] = pd.to_numeric(df['MODIS-PWV(mm)'], errors='coerce')

# Perform t-test
# If there are still issues, you may want to check for constant values or data size
try:
    t_statistic, p_value = ttest_ind(df['GNSS-DERIVED PWV'], df['MODIS-PWV(mm)'])
    # Print results
    print(f'T-statistic: {t_statistic}\nP-value: {p_value}')
except Exception as e:
    print(f"Error: {e}")

# Display the DataFrame to inspect the data
print(df)

# Interpret results
if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")


# In[39]:


#2015
import pandas as pd
from scipy.stats import ttest_ind

# Load data from Excel file
# Replace 'ANOVA FOR 2015.xlsx' with the actual name of your Excel file
df = pd.read_excel('T-TEST FOR 2015.xlsx')

# Check for missing values and handle them if needed
if df.isnull().sum().any():
    # If there are missing values, you can choose to drop or fill them
    df = df.dropna()  # Use df.fillna() if you want to fill missing values

# Ensure numeric data types for 'GNSS-DERIVED PWV' and 'MODIS-PWV(mm)'
df['GNSS-DERIVED PWV'] = pd.to_numeric(df['GNSS-DERIVED PWV'], errors='coerce')
df['MODIS-PWV(mm)'] = pd.to_numeric(df['MODIS-PWV(mm)'], errors='coerce')

# Perform t-test
# If there are still issues, you may want to check for constant values or data size
try:
    t_statistic, p_value = ttest_ind(df['GNSS-DERIVED PWV'], df['MODIS-PWV(mm)'])
    # Print results
    print(f'T-statistic: {t_statistic}\nP-value: {p_value}')
except Exception as e:
    print(f"Error: {e}")

# Display the DataFrame to inspect the data
print(df)

# Interpret results
if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")


# In[29]:


#2016
import pandas as pd
from scipy.stats import ttest_ind

# Load data from Excel file
# Replace 'ANOVA FOR 2015.xlsx' with the actual name of your Excel file
df = pd.read_csv('TIMESERIES2016.csv')

# Check for missing values and handle them if needed
if df.isnull().sum().any():
    # If there are missing values, you can choose to drop or fill them
    df = df.dropna()  # Use df.fillna() if you want to fill missing values

# Ensure numeric data types for 'GNSS-DERIVED PWV' and 'MODIS-PWV(mm)'
df['GNSS-derived PWV'] = pd.to_numeric(df['GNSS-derived PWV'], errors='coerce')
df['MODIS-PWV(mm)'] = pd.to_numeric(df['MODIS-PWV(mm)'], errors='coerce')

# Perform t-test
# If there are still issues, you may want to check for constant values or data size
try:
    t_statistic, p_value = ttest_ind(df['GNSS-derived PWV'], df['MODIS-PWV(mm)'])
    # Print results
    print(f'T-statistic: {t_statistic}\nP-value: {p_value}')
except Exception as e:
    print(f"Error: {e}")

# Display the DataFrame to inspect the data
print(df)

# Interpret results
if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")


# In[28]:


#2017
import pandas as pd
from scipy.stats import ttest_ind

# Load data from Excel file
# Replace 'ANOVA FOR 2015.xlsx' with the actual name of your Excel file
df = pd.read_csv('TIMESERIES2017.csv')

# Check for missing values and handle them if needed
if df.isnull().sum().any():
    # If there are missing values, you can choose to drop or fill them
    df = df.dropna()  # Use df.fillna() if you want to fill missing values

# Ensure numeric data types for 'GNSS-DERIVED PWV' and 'MODIS-PWV(mm)'
df['GNSS-derived PWV'] = pd.to_numeric(df['GNSS-derived PWV'], errors='coerce')
df['MODIS-PWV'] = pd.to_numeric(df['MODIS-PWV'], errors='coerce')

# Perform t-test
# If there are still issues, you may want to check for constant values or data size
try:
    t_statistic, p_value = ttest_ind(df['GNSS-derived PWV'], df['MODIS-PWV'])
    # Print results
    print(f'T-statistic: {t_statistic}\nP-value: {p_value}')
except Exception as e:
    print(f"Error: {e}")

# Display the DataFrame to inspect the data
print(df)

# Interpret results
if p_value < 0.05:
    print("Reject the null hypothesis. There is a significant difference.")
else:
    print("Fail to reject the null hypothesis. There is no significant difference.")


# In[38]:


import pandas as pd
import scipy.stats as stats

# Load data from Excel file
# Replace 'ANOVA FOR 2015.2.xlsx' with the actual name of your Excel file
df = pd.read_excel('T-TEST FOR 2015.xlsx')

# Assume your Excel file has two columns: 'GNSS-DERIVED PWV' and 'MODIS-PWV(mm)'
# Replace these with your actual column names
sample1 = df['GNSS-DERIVED PWV']
sample2 = df['MODIS-PWV(mm)']

# Check for missing values
if sample1.isnull().any() or sample2.isnull().any():
    print("Warning: Missing values detected. Handle or remove them before proceeding.")
else:
    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(sample1, sample2)

    # Print the results
    print("Independent Samples T-Test:")
    print(f"T-statistic: {t_statistic}")
    print(f"P-value: {p_value}")

    # Interpret the results
    if p_value < 0.05:
        print("Reject the null hypothesis. There is a significant difference.")
    else:
        print("Fail to reject the null hypothesis. There is no significant difference.")


# In[ ]:




