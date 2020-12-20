import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd, Week
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
stocks_week = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Convert to CRSP and Compustat dates to date time objects
stocks_week['DATE'] = pd.to_datetime(stocks_week['DATE'], format = "%Y-%m-%d")
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Introduce year and variables
stocks_week['year'] = stocks_week['DATE'].dt.year
stocks_week['month'] = stocks_week['DATE'].dt.month

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
stocks_month['year'] = faux_date.dt.year
stocks_month['month'] = faux_date.dt.month

del faux_date

# Select columns for future merge
columns = stocks_month[['year', 'month', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY', 'PRICE_UNADJUSTED', 'TOTAL_SHARES', 'TOT_EQUITY']]

# No more need for monthly data set
del stocks_month 

# Merge with weekly data
stocks_week = stocks_week.merge(columns, on = ['year', 'month', 'DW_INSTRUMENT_ID'])

# subset to needed columns
stocks_week = stocks_week[['DATE', 'year', 'month', 'DW_INSTRUMENT_ID', 'TOTAL_SHARES', 'MAIN_KEY', 'INDUSTRY', 'RETURN', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]

# Merge with clusters
stocks_week = stocks_week.merge(clusters, on = ['year', 'MAIN_KEY'])

# Only consider observations before the year 2010
stocks_week = stocks_week.loc[stocks_week['year'] < 2010, :]

# Convert returns to decimals
stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']

# Calculate fundemental return; equal weighted within cluster
stocks_week['fund'] = stocks_week.groupby(['DATE', 'cluster'])['RETURN'].transform('mean')

# Calculate faux SIC return; equal weighted within each faux SIC
stocks_week['ind'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Calculate official minus fundemental
stocks_week['ind-fund'] = stocks_week['ind'] - stocks_week['fund']

# Create industry and cluster counts
stocks_week['INDUSTRY_n'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
stocks_week['cluster_n'] = stocks_week.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')

# Calculate total market equity
stocks_week['TOTAL_MCAP'] = stocks_week['PRICE_UNADJUSTED'] * stocks_week['TOTAL_SHARES']

# Log of total market capitalization
stocks_week['ln_TOTAL_MCAP'] = np.log(stocks_week['TOTAL_MCAP'])

# Calculate log of book to market equity
stocks_week['ln_BMR'] = np.log(stocks_week['TOT_EQUITY']/stocks_week['TOTAL_MCAP'])

# Sort values for shift
stocks_week.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Record valid shifts
stocks_week['valid'] = stocks_week['DATE'].shift(-1)== stocks_week['DATE'] + dt.timedelta(days = 7)

# Perform shift on returns
stocks_week['RETURN_LEAD'] = stocks_week.groupby('DW_INSTRUMENT_ID')['RETURN'].shift(-1)

# Remove invaluds
stocks_week.loc[stocks_week['valid'] == False, 'RETURN_LEAD'] = np.nan

# Shift data one week forward
stocks_week['DATE'] = stocks_week['DATE'] + dt.timedelta(days = 7)

# Redefine month and date columns
stocks_week['year'] = stocks_week['DATE'].dt.year
stocks_week['month'] = stocks_week['DATE'].dt.month

# Construct function to calculate cumlative returns
prod = lambda x: np.prod(1 + x) - 1

# Define valid
stocks_week['valid'] = stocks_week['DATE'].shift(11) + dt.timedelta(days = 7 * 11) == stocks_week['DATE']

# How much time has it been so far?
print(0.5 * int((time.time() - start_time)/30), 'minutes so far...')

# Calculate cumative returns
stocks_week['CUM_RETURN'] = stocks_week.groupby('DW_INSTRUMENT_ID')['RETURN'].transform(lambda x: x.shift(1).rolling(10).apply(prod))

# Calculate after 2002
stocks_week['after_2002'] = (stocks_week['DATE'].dt.year > 2002).astype(int)

# Only consider firms with more than 5 completing industries based both on SIC and cluster
good_firm = (stocks_week['INDUSTRY_n'] > 4) & (stocks_week['cluster_n'] > 4) & (stocks_week['PRICE_UNADJUSTED'] >= 5) & (stocks_week['TOT_EQUITY'] > 0)
stocks_week = stocks_week.loc[good_firm, :]

# Drop unneeded columns
stocks_week.drop(['fund', 'ind', 'INDUSTRY_n', 'cluster_n', 'valid', 'TOTAL_SHARES', 'MAIN_KEY', 'PRICE_UNADJUSTED', 'TOT_EQUITY'], axis = 1, inplace = True)

# Drop observations with nan values
stocks_week.dropna(axis = 0, inplace = True)

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
    
    temp = stocks_week.groupby(groupby).apply(lambda x: get_pars(x, variables = vars)).reset_index()

    vars.insert(0, 'const')
    
    betas = pd.DataFrame(temp[0].tolist(), columns = vars)
    
    del temp
    
    # Only concerned with the alpha and the factor loadings for the variables, not the dummies
    betas = betas.iloc[:, 0:(len(vars) + 1)]

    results =  pd.DataFrame(columns = ['betas', 't-values'], index = vars)   
    
    # Take the mean to obtain beta estimate over time
    results['betas'] = betas.mean(axis = 0).tolist()

    # Use the central limit theorem and the standard devation of betas to estimate the SE
    SE = betas.std(axis = 0)/np.sqrt(len(betas))
    
    # Use SE to obtain t-values 
    results['t-values'] = results['betas']/SE
    
    return results
    
# Create and save results
today = str(pd.to_datetime("today"))[0:10]

vars_list = [['ind-fund'], ['ind-fund', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN'], ['ind-fund', 'RETURN', 'CUM_RETURN'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR', 'after_2002']]
groupby = ['year', 'month']

writer = pd.ExcelWriter(path + '\\Data\\clusters_table7_results\\table7_results_weekly_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

i = 1

for vars in vars_list:
    
    convert_betas(groupby = groupby, vars = vars).to_excel(writer, sheet_name = str(i))
    i += 1

writer.save()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')
