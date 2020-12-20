import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
import datetime as dt
import time

# Record start time
start_time = time.time()

# Record path
path = r'C:\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What type of linkage?
linkage = 'complete'

# What is the cut?
cut = 0.15

# Load clusters 
clusters = pd.read_csv(path + '\\Data\\Clusters\\clusters_cut_' + str(cut) + '_linkage_' + linkage + ".csv")

# Hoberg score in year Y is matched to stock data in year Y + 1
clusters['year'] = 1 + clusters['year']

# Load return data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Convert to CRSP and Compustat dates to date time objects
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Introduce year variable
stocks_month['year'] = stocks_month['DATE'].dt.year

# Introduce month variable
stocks_month['month'] = stocks_month['DATE'].dt.month

# subset to needed columns
stocks_month = stocks_month.loc[:, ['DATE', 'year', 'month', 'DW_INSTRUMENT_ID', 'TOTAL_SHARES', 'MAIN_KEY', 'INDUSTRY', 'RETURN', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]

# Merge with clusters
stocks_month = stocks_month.merge(clusters, on = ['year', 'MAIN_KEY'])

# Only consider observations before the year 2010
stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]

# Convert returns to decimals
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']

# Calculate fundemental return; equal weighted within cluster
stocks_month['fund'] = stocks_month.groupby(['DATE', 'cluster'])['RETURN'].transform('mean')

# Calculate faux SIC return; equal weighted within each faux SIC
stocks_month['ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Calculate official minus fundemental
stocks_month['ind-fund'] = stocks_month['ind'] - stocks_month['fund']

# Create SIC and cluster counts
stocks_month['INDUSTRY_n'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
stocks_month['cluster_n'] = stocks_month.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')

# Calculate total market equity
stocks_month['TOTAL_MCAP'] = stocks_month['PRICE_UNADJUSTED'] * stocks_month['TOTAL_SHARES']

# Log of total market capitalization
stocks_month['ln_TOTAL_MCAP'] = np.log(stocks_month['TOTAL_MCAP'])

# Calculate log of book to market equity
stocks_month['ln_BMR'] = np.log(stocks_month['TOT_EQUITY']/stocks_month['TOTAL_MCAP'])

# Sort values for shift
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Record valid shifts
stocks_month['valid'] = stocks_month['DATE'].shift(-1) + MonthEnd(0) == stocks_month['DATE'] + dt.timedelta(days = 7)  + MonthEnd(0) 

# Perform shift on returns
stocks_month['RETURN_LEAD'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].shift(-1)

# Shift dates forward by one month
stocks_month['DATE'] = stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0)

# Redefine month and date columns
stocks_month['year'] = stocks_month['DATE'].dt.year
stocks_month['month'] = stocks_month['DATE'].dt.month

# Remove invaluds
stocks_month.loc[stocks_month['valid'] == False, 'RETURN_LEAD'] = np.nan

# Construct function to calculate cumlative returns
prod = lambda x: np.prod(1 + x) - 1

# Define valid
stocks_month['valid'] = stocks_month['DATE'].shift(11) + dt.timedelta(days = 7) + MonthEnd(11) == stocks_month['DATE'] + MonthEnd(0)

# Calculate cumative returns
stocks_month['CUM_RETURN'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].transform(lambda x: x.shift(1).rolling(10).apply(prod))

# Calculate after 2002
stocks_month['after_2002'] = (stocks_month['DATE'].dt.year > 2002).astype(int)

# Only consider firms with more than 5 completing industries based both on SIC and cluster
good_firm = (stocks_month['INDUSTRY_n'] > 4) & (stocks_month['cluster_n'] > 4) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['TOT_EQUITY'] > 0)
stocks_month = stocks_month.loc[good_firm, :]

# Drop unneeded columns
stocks_month.drop(['fund', 'ind', 'INDUSTRY_n', 'cluster_n', 'valid', 'TOTAL_SHARES', 'MAIN_KEY', 'PRICE_UNADJUSTED', 'TOT_EQUITY'], axis = 1, inplace = True)

# Drop observations with nan values
stocks_month.dropna(axis = 0, inplace = True)

def get_pars(data, variables):
    
    # Add returns at t + 1 to varaibles
    variables.append('RETURN_LEAD')
    
    # Get month dummies
    dummies = pd.get_dummies(data['month'])
    
    # Concert columns to numeric
    data = data[variables].apply(pd.to_numeric)
    
    # Drop rows with missing values
    data.dropna(axis = 0, inplace = True)
    
    # Remove RETURN
    variables.remove('RETURN_LEAD')
    
    # Choose the columns of stocks_month for regression; add column of ones
    X = sm.add_constant(pd.concat([data[variables], dummies], axis = 1).values)
    
    # Save response variable
    y = data['RETURN_LEAD'].values

    try:
        # Peform the regression
        res = sm.OLS(y, X).fit()
        
        return res.params
    
    except:
        
        return np.repeat(np.nan, len(variables))

def convert_betas(groupby, vars):
    
    temp = stocks_month.groupby(groupby).apply(lambda x: get_pars(x, variables = vars)).reset_index()

    vars.insert(0, 'const')
    
    betas = pd.DataFrame(temp[0].tolist(), columns = vars)
    
    del temp
    
    # Only concerned with the alpha and the factor loadings for the variables, not the dummies
    betas = betas.iloc[:, 0:(len(vars) + 1)]

    results =  pd.DataFrame(columns = ['betas', 't-values'], index = vars)   
    
    # Take the mean to obtain beta estimate over time
    results['betas'] = betas.mean(axis = 0).tolist()

    # Use the central limit theorem and the standard devation of betas to estimate the SE
    SE = (betas.std(axis = 0)/np.sqrt(len(betas)))
    
    # Use SE to obtain t-values 
    results['t-values'] = results['betas']/SE
    
    return results
    
    # Create and save results
today = str(pd.to_datetime("today"))[0:10]

vars_list = [['ind-fund'], ['ind-fund', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN'], ['ind-fund', 'RETURN', 'CUM_RETURN'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR', 'after_2002']]
groupby = ['year', 'month']

writer = pd.ExcelWriter(path + '\\Data\\clusters_table7_results\\table7_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

i = 1

for vars in vars_list:
    
    convert_betas(groupby = groupby, vars = vars).to_excel(writer, sheet_name = str(i))
    
    i += 1

writer.save()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')
