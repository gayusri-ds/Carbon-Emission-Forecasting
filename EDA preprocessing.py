# -*- coding: utf-8 -*-


# CARBON EMISSION DATA PREPROCESSING

# IMPORTING THE REQUIRED LIBRARIES

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine.outliers import Winsorizer

import sweetviz

from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


# Importing the dataset 

df = pd.read_csv(r"C:/Users/DELL/Desktop/My Projects/Emission/emissions.csv")



# Extarct data points which are all above 45 in each unique combinations
# Group the data by 'state-name', 'sector-name', and 'fuel-name'
grouped = df.groupby(['state-name', 'sector-name', 'fuel-name'])

# Filter groups with at least 45 data points
data = grouped.filter(lambda x: len(x) >= 45)

###################################################################################################################


# EDA (EXPLORATORY DATA ANALYSIS)


# Checking the null values
data.isnull().sum()  # No null values present


# Checking the duplicate values
data.duplicated().sum()  # No duplicate values in the dataset


# Basic information of the dataset
data.info()  # Basic information


# Basic statistics of teh dataset
data.describe()  # Basic statistics

# Data types
data.dtypes


#############################################################################################################

# AutoEDA(Automated Exploratory Data Analysis)


report = sweetviz.analyze([data, 'data'])
report.show_html('My_Report.html')


#############################################################################################################

# Exploratory Data Analysis

# First moment Business decision( measure of central tendency)
# MEAN
mean_sep = data.groupby(['state-name', 'sector-name','fuel-name'])['value'].mean().reset_index()


# MEDIAN
median_sep = data.groupby(['state-name','sector-name','fuel-name'])['value'].median().reset_index()


# MODE
mode_sep = data.groupby(['state-name','sector-name','fuel-name'])['value'].agg(lambda x : x.mode().iloc[0] if not x.mode().empty else None).reset_index()


# Second  Moment Business decision
# STANDARD DEVIATION
std_sep = data.groupby(['state-name','sector-name','fuel-name'])['value'].std().reset_index()


# VARIANCE
var_sep = data.groupby(['state-name','sector-name','fuel-name'])['value'].var().reset_index()


# RANGE
range_sep = data.groupby(['state-name', 'sector-name', 'fuel-name'])['value'].agg(lambda x : x.max() - x.min()).reset_index()


# Third Moment Business Decision
# Skewness
skew_sep = data.groupby(['state-name', 'sector-name', 'fuel-name'])['value'].skew().reset_index()


#Fourth Moment Business Decision
# Kurtosis
kurt_sep = data.groupby(['state-name', 'sector-name', 'fuel-name'])['value'].agg(lambda x : x.kurt()).reset_index()

###############################################################################################################################################################################



# Checking for OUTLIERS using Boxplot Visulaisation

data.plot(kind = 'box', subplots = True,sharey = False, figsize = (16,8))



# Using WINSORIZAION method for outlier treatment
winsor = Winsorizer(capping_method = 'iqr', fold = 1.5, tail = 'both', variables = ['value'])

win =   winsor.fit_transform(data[['value']])

data['value'] = win

data.plot(kind = 'box', subplots = True, sharey = False, figsize = (20,10))

# All outlier values are treated


##################################################################################################################

# UNIQUE COMBINATIONS of data 

unique_combination = data[['state-name', 'sector-name', 'fuel-name']].drop_duplicates()



# Basic visualization 
# Iterate through unique combinations to plot line charts
for index, row in unique_combination.iterrows():
    state = row['state-name']
    sector = row['sector-name']
    fuel = row['fuel-name']
    
    # Filter data for the current combination
    filtered_df = data[(data['state-name'] == state) & (data['sector-name'] == sector) & (data['fuel-name'] == fuel)]
    
    # Plot line chart for the current combination
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df['year'], filtered_df['value'], marker='o', label=f'{state}, {sector}, {fuel}')
    plt.title(f'Emissions for {state}, {sector}, {fuel}')
    plt.xlabel('Year')
    plt.ylabel('Emissions Value')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    
    
################################################################################################################    

# Check dataset is NORMAL OR NOTNORMAL

# SHAPIRO TEST 
# SHAPIRO TEST FOR WHOLE DATASET (NORMAL OR NOT NORMAL)
# Extract emissions data
emissions_data = data['value'].tolist()

# Perform Shapiro-Wilk test
statistic, p_value = shapiro(emissions_data)

# Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(emissions_data, bins=5, color='blue', alpha=0.7)
plt.title('Histogram of Emissions Data')
plt.xlabel('Emissions Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Print Shapiro-Wilk test results
print(f"Shapiro-Wilk Test Statistic: {statistic}")
print(f"P-value: {p_value}")

# Interpret the test results
alpha = 0.05
if p_value > alpha:
    print("\nThe data follows a normal distribution (fail to reject H0)")
else:
    print("\nThe data does not follow a normal distribution (reject H0)")  # Not a NORMAL DISTRIBUTION 


###################################################################################################################



# SHAPIRO TEST FOR UNIQUE COMBINATIONS OF DATASET SEPARATELY 

# Dictionary to store Shapiro-Wilk test results
shapiro_results = {
    'state-name': [],
    'sector-name': [],
    'fuel-name': [],
    'shapiro_test_result': []
}

# Loop through each unique combination
for idx, row in unique_combination.iterrows():
    state, sector, fuel = row['state-name'], row['sector-name'], row['fuel-name']
    
    # Filter data for the current combination
    filtered_df = data[(data['state-name'] == state) & (data['sector-name'] == sector) & (data['fuel-name'] == fuel)]
    
    # Extract emissions data
    emissions_data = filtered_df['value'].tolist()
    
    # Check data length and range
    if len(emissions_data) >= 3 and max(emissions_data) > min(emissions_data):
        # Perform Shapiro-Wilk test
        statistic, p_value = shapiro(emissions_data)
        
        # Store test results
        shapiro_results['state-name'].append(state)
        shapiro_results['sector-name'].append(sector)
        shapiro_results['fuel-name'].append(fuel)
        shapiro_results['shapiro_test_result'].append('Normal' if p_value > 0.05 else 'Not Normal')
    else:
        # Store 'Insufficient Data' for combinations with less than 3 data points or zero range
        shapiro_results['state-name'].append(state)
        shapiro_results['sector-name'].append(sector)
        shapiro_results['fuel-name'].append(fuel)
        shapiro_results['shapiro_test_result'].append('Insufficient Data')

# Create DataFrame from results
normality_result  = pd.DataFrame(shapiro_results)


#################################################################################################################

# Number of random walk and not a random walk present in the dataset  



# List to store results
random_walk_count = 0
non_random_walk_count = 0

# Loop through each time series
for idx, row in unique_combination.iterrows():
    state, sector, fuel = row['state-name'], row['sector-name'], row['fuel-name']
    
    # Filter data for the current combination
    filtered_df = data[(data['state-name'] == state) & 
                       (data['sector-name'] == sector) & 
                       (data['fuel-name'] == fuel)]
    
    # Extract time series data
    time_series = filtered_df['value']
    
    # Perform ADF test
    result_adf = adfuller(time_series)
    p_value_adf = result_adf[1]
    
    # Check if the time series is a random walk
    if p_value_adf > 0.05:
        random_walk_count += 1
    else:
        non_random_walk_count += 1

print("Number of time series exhibiting random walk behavior:", random_walk_count)
print("Number of time series not exhibiting random walk behavior:", non_random_walk_count)



###########################################################################################################

# Only use NOT A RANDOM WALK data points

# List to store not random walk time series data
not_random_walk_data = []

# Loop through each time series
for idx, row in unique_combination.iterrows():
    state, sector, fuel = row['state-name'], row['sector-name'], row['fuel-name']
    
    # Filter data for the current combination
    filtered_df = data[(data['state-name'] == state) & 
                       (data['sector-name'] == sector) & 
                       (data['fuel-name'] == fuel)]
    
    # Extract time series data
    time_series = filtered_df['value']
    
    # Perform ADF test
    result_adf = adfuller(time_series)
    p_value_adf = result_adf[1]
    
    # Check if the time series is not a random walk (p-value < 0.05)
    if p_value_adf < 0.05:
        # Append the time series data to the list
        not_random_walk_data.append(filtered_df)

# Concatenate the filtered dataframes to create a single dataframe
not_random_walk_df = pd.concat(not_random_walk_data)

# Print the shape of the dataframe
print("Shape of the dataframe containing not random walk data:", not_random_walk_df.shape)

new = not_random_walk_df

# Save the data into sepaarte csv file 

new.to_csv(r"Desktop/not_outlier.csv")


###########################################################################################################


# List of all unique combinations
unique_combinations = new[['state-name', 'sector-name', 'fuel-name']].drop_duplicates()

# Initialize empty lists to store results

stationary_data = []
non_stationary_data = []

# Loop over each unique combination
for index, row in unique_combinations.iterrows():
    state = row['state-name']
    sector = row['sector-name']
    fuel = row['fuel-name']
    
    print(f"\nChecking stationarity for {state}, {sector}, {fuel}:")
    
    # Filter data for the current combination
    filtered_data = new[(new['state-name'] == state) & 
                         (new['sector-name'] == sector) & 
                         (new['fuel-name'] == fuel)]
    
    # Take the time series column 'value'
    time_series = filtered_data['value']
    
    # Perform ADF Augmented Dickey-Fuller test
    result_adf = adfuller(time_series)
    p_value_adf = result_adf[1]
    
    # Perform KPSS test
    result_kpss = kpss(time_series)
    p_value_kpss = result_kpss[1]
    
    # Check stationarity based on p-values for ADF and KPSS tests
    if p_value_adf < 0.05 and p_value_kpss > 0.05:
        adf_status = 'Stationary'
        random_walk_status = 'Not Random Walk'
        stationary_data.append(filtered_data)
    else:
        adf_status = 'Not Stationary'
        random_walk_status = 'Random Walk'
        non_stationary_data.append(filtered_data)
    
    # Perform AR Lagged Test for random walk
    lag_data = time_series.shift().dropna()
    lagged_data = sm.add_constant(lag_data)
    try:
        model = sm.OLS(time_series[1:], lagged_data)
        result_lag = model.fit()
        
        # Get the coefficients
        coefficients = result_lag.params
        
        # Check if the constant term exists in the coefficients
        if 'const' in coefficients.index:
            lag_coefficient = coefficients['const']
            
            # Check if the coefficient is close to 1 for a random walk
            if round(lag_coefficient, 2) == 1.0:
                random_walk_status = 'Random Walk'
    except Exception as e:
        print(f"Error in AR Lagged Test for {state}, {sector}, {fuel}: {e}")
        

# Concatenate the stationary and non-stationary data
stationary_data = pd.concat(stationary_data)
non_stationary_data = pd.concat(non_stationary_data)

# Save the stationary and non-stationary data into separate CSV files

stationary_data.to_csv(r'C:\Users\DELL\Desktop\My Projects\Emission\stationary_data_30.csv', index=False)
non_stationary_data.to_csv(r'C:\Users\DELL\Desktop\My Projects\Emission\non_stationary_data_30.csv', index=False)

########################################################################################################################################









