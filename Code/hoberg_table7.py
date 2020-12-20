import pandas as pd
import numpy as np
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
import datetime as dt
import time
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path
path = r'C:\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009?
post_2009 = False

# Where are the fundamental returns
file = path + '\\Data\\fundamental_mods\\monthly_fund_rtn_cluster_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = 'fund_rtn_clusters'

# Get the month return data
stocks_month = cf.prep_monthly_rtn(path, cut, file, post_2009)

# Calculate faux SIC return; equal weighted within each faux SIC
stocks_month['ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Calculate official minus fundemental
stocks_month['ind-fund'] = stocks_month['ind'] - stocks_month['TNIC_RET']

# Create SIC and cluster counts
stocks_month['INDUSTRY_N'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')

# Calculate total market equity
stocks_month['TOTAL_MCAP'] = stocks_month['PRICE_UNADJUSTED'] * stocks_month['TOTAL_SHARES']

# Log of total market capitalization
stocks_month['ln_TOTAL_MCAP'] = np.log(stocks_month['TOTAL_MCAP'])

# Calculate log of book to market equity
stocks_month['ln_BMR'] = np.log(stocks_month['TOT_EQUITY']/stocks_month['TOTAL_MCAP'])

# Sort values for shift
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Record valid shifts
stocks_month['valid'] = stocks_month['DATE'].shift(-1) + MonthEnd(0) == stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) 

# Perform shift on returns
stocks_month['RETURN_LEAD'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].shift(-1)

# Shift dates forward one month
stocks_month['DATE'] = stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0)

# Redefine month and date columns
stocks_month['year'] = stocks_month['DATE'].dt.year
stocks_month['month'] = stocks_month['DATE'].dt.month

# Remove invaluds
stocks_month.loc[stocks_month['valid'] == False, 'RETURN_LEAD'] = np.nan

# Construct function to calculate cumlative returns
prod = lambda x: (1 + x).prod() - 1

# Define valid
stocks_month['valid'] = stocks_month['DATE'].shift(11) + dt.timedelta(days = 7) + MonthEnd(11) == stocks_month['DATE'] + MonthEnd(0)

# Calculate cumative returns
stocks_month['CUM_RETURN'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].transform(lambda x: x.shift(1).rolling(10).apply(prod))

# Calculate after 2002
stocks_month['after_2002'] = (stocks_month['DATE'].dt.year > 2002).astype(int)

# Only consider firms with more than 5 completing industries based both on industry and hoberg, price >= 5, and positive book equity
good_firm = (stocks_month['INDUSTRY_N'] > 4) & (stocks_month['N'] > 4) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['TOT_EQUITY'] > 0)
stocks_month = stocks_month.loc[good_firm, :]

# Place firms into terciles based on market capitilizatin
stocks_month['cap_quant'] = stocks_month.groupby('DATE')['TOTAL_MCAP'].transform(lambda x: pd.qcut(x, 3, duplicates = 'drop', labels = False))

# Get dummy variables for market cap
stocks_month[['I_1', 'I_2', 'I_3']] = pd.get_dummies(stocks_month['cap_quant'])

# Create variables for interactions
for var in ['ind', 'TNIC_RET', 'ind-fund']:
    
    stocks_month[var +'*I_1'] = stocks_month[var] * stocks_month['I_1']
    stocks_month[var +'*I_3'] = stocks_month[var] * stocks_month['I_3']

# Drop unneeded columns
stocks_month.drop(['INDUSTRY_N', 'I_2', 'N', 'valid', 'cap_quant', 'TOTAL_MCAP', 'TOTAL_SHARES', 'PRICE_UNADJUSTED', 'TOT_EQUITY'], axis = 1, inplace = True)

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
    
    betas = betas.iloc[:, 0:(len(vars) + 1)]

    results =  pd.DataFrame(columns = ['betas', 't-values'], index = vars)   
    
    results['betas'] = betas.mean(axis = 0).tolist()

    results['t-values'] = results['betas']/(betas.std(axis = 0)/np.sqrt(len(betas)))
    
    return results
    
    # Create and save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\hoberg_table7_results\\'
else:
    location = path + '\\Data\\' + folder + '\\'

vars_list = [['ind-fund'], ['ind-fund', 'ind-fund*I_1', 'ind-fund*I_3'], ['ind-fund', 'ind-fund*I_1', 'ind-fund*I_3', 'after_2002'],
             ['TNIC_RET'], ['TNIC_RET', 'TNIC_RET*I_1', 'TNIC_RET*I_3'], ['TNIC_RET', 'TNIC_RET*I_1', 'TNIC_RET*I_3', 'after_2002'],
             ['ind'], ['ind', 'ind*I_1', 'ind*I_3'], ['ind', 'ind*I_1', 'ind*I_3', 'after_2002'],
             ['ind-fund', 'RETURN'], ['ind-fund', 'RETURN', 'ind-fund*I_1', 'ind-fund*I_3'], ['ind-fund', 'RETURN', 'ind-fund*I_1', 'ind-fund*I_3', 'after_2002'], 
             ['TNIC_RET', 'ind'], ['TNIC_RET', 'TNIC_RET*I_1', 'TNIC_RET*I_3', 'ind*I_1', 'ind*I_3'], 
             ['ind-fund', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN', 'CUM_RETURN'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'after_2002'], 
             ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'after_2002'], 
             ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR'], ['ind-fund', 'RETURN', 'CUM_RETURN', 'ln_TOTAL_MCAP', 'ln_BMR', 'after_2002']]
groupby = ['year', 'month']

if folder is None:
    if post_2009 == True:
        writer = pd.ExcelWriter(location + 'post_2009_table7_results_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        writer = pd.ExcelWriter(location + 'table7_results_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    if post_2009 == True:
        writer = pd.ExcelWriter(location + 'Table_7\\post_2009_table7_results_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        writer = pd.ExcelWriter(location + 'Table_7\\table7_results_cut_' + str(cut) + '_' + today + '.xlsx')
    
i = 1

for vars in vars_list:
    
    convert_betas(groupby = groupby, vars = vars).to_excel(writer, sheet_name = str(i))
    
    i += 1

writer.save()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes')