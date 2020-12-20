import pandas as pd
import datetime as dt
import numpy as np
from pandas.tseries.offsets import MonthEnd
import copy
import time
import matplotlib.pyplot as plt
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009?
post_2009 = True

# Where are the fundamental returns?
file = path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = None

# Load weekly return data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Convert dates to datetime objects
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

if post_2009 == True:
    # Only consider observations after 2009
    stocks_month = stocks_month.loc[stocks_month['DATE'].dt.year > 2009, :] 
else:
    # Only consider observations up to 2009
    stocks_month = stocks_month.loc[stocks_month['DATE'].dt.year < 2010, :]

# Create market equity column
stocks_month['ME'] = stocks_month['PRICE_UNADJUSTED'] * stocks_month['TOTAL_SHARES']

# Sort columns
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Shift market equity
stocks_month['ME_Lag'] = stocks_month.groupby('DW_INSTRUMENT_ID')['ME'].shift(1)

# Create flag for invalids
stocks_month['valid'] = stocks_month.groupby('DW_INSTRUMENT_ID')['DATE'].shift(1) + dt.timedelta(days = 7) + MonthEnd(0) == stocks_month['DATE'] + MonthEnd(0)

# Remove the invalid observations
stocks_month.loc[stocks_month['valid'] == False, 'ME_Lag'] = np.nan

# Break into terciles by market equity
stocks_month['quant'] = stocks_month.groupby(['DATE'])['ME_Lag'].transform(lambda x: pd.qcut(x, 3, duplicates = 'drop', labels = False))

# Drop columns that have done their jobs
stocks_month.drop(['valid', 'ME', 'ME_Lag'], axis = 1, inplace = True)

# Drop duplicates
stocks_month.drop_duplicates(inplace = True)

# Drop values missing instrument ID or gvkey
stocks_month.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert MAIN_KEY to int64
stocks_month['MAIN_KEY'] = stocks_month['MAIN_KEY'].astype('int64')

# Load fundamental return data
if file == None:   
    hoberg = pd.read_csv(path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv')
else:
    hoberg = pd.read_csv(file)

# Convert data column to date obect
hoberg['DATE'] = pd.to_datetime(hoberg['DATE'], format = "%Y-%m-%d")

# Fundamental and return data
stocks_month = stocks_month.merge(hoberg, on = ['DATE', 'MAIN_KEY', 'DW_INSTRUMENT_ID'])

del hoberg

# Convert returns to a decimal
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']
stocks_month['TNIC_RET'] = 0.01 * stocks_month['TNIC_RET'] 

# Shift date to month end for merge with FF later
stocks_month['DATE'] = stocks_month['DATE'] + MonthEnd(0)

# Load Fama French three factors
FF, RF = cf.prep_FF_monthly(path)

# Get columns for right side of merge
large = stocks_month.loc[stocks_month['quant'] == 2, :]

# Drop quant
large.drop('quant', axis = 1, inplace = True)

# Only consider the smallest stocks measured by ME
small = stocks_month.loc[stocks_month['quant'] == 0, :]

# Drop quant
small.drop('quant', axis = 1, inplace = True)

del stocks_month

# Construct function to do Hoberg calculation
def calculate_stuff(data, grouping):
    
    df = copy.deepcopy(data)
    
    if (grouping == 'INDUSTRY')|(grouping == 'ind-fund'):
        
        df.dropna(subset = ['INDUSTRY'], axis = 0, inplace = True)
        
        df['INDUSTRY'] = df.groupby('INDUSTRY')['RETURN'].transform('mean')
        
        df['N_ind'] = df.groupby('INDUSTRY')['DW_INSTRUMENT_ID'].transform('count')
        
        if grouping == 'ind-fund':        
            df['ind-fund']  =  df['INDUSTRY'] -  df['TNIC_RET']
        else:
            df['N'] = 5    
    else:
        df['N_ind'] = 5
    

    # Shift fund back one week
    df[grouping] = df.groupby('DW_INSTRUMENT_ID')[grouping].shift(1)

    # Record valids
    df['valid'] = df.groupby(['DW_INSTRUMENT_ID'])['DATE'].shift(1) + dt.timedelta(days = 7) + MonthEnd(0) == df['DATE'] + MonthEnd(0)

    # Remove invalids
    df.loc[df['valid'] == False, grouping] = np.nan

    # Drop the usual suspects
    good_firm = (df['TOT_EQUITY'] > 0) & (df['PRICE_UNADJUSTED'] >= 5) & (df['N'] > 2) & (df['N_ind'] > 2)
    df = df.loc[good_firm, :]
    
    # Drop missing values
    df.dropna(subset = [grouping], axis = 0, inplace = True)

    # Create quintiles
    df['quant'] = df.groupby('DATE')[grouping].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))

    # Take the equally weighted return of each quantile
    df = df.groupby(['DATE', 'quant'])['RETURN'].mean().reset_index()

    # Restructure dataframe
    df = df.pivot(index = 'DATE', columns = 'quant', values = 'RETURN').reset_index()

    # Rename columns
    df.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace = True)
    
    # When Q5 is missing, replace with Q4
    df.loc[df['Q5'].isna(), 'Q5'] = df.loc[df['Q5'].isna(), 'Q4']
    
    # Merge with risk-free
    df = df.merge(RF, on = 'DATE')
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        df[Q + '-RF'] = df[Q] - df['RF']

    # Calculate Q1-Q5
    df['Q1-Q5'] = df['Q1'] - df['Q5']    
        
    # Calculate Q5-Q1
    df['Q5-Q1'] = df['Q5'] - df['Q1']
    
    return df

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Predict the returns of small firms using industry returns
small_by_ind = calculate_stuff(small, 'INDUSTRY')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Predict the returns of small firms using fundamental returns
small_by_fund = calculate_stuff(small, 'TNIC_RET')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Predict the returns of small firms by the difference
small_by_diff = calculate_stuff(small, 'ind-fund')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Predict the returns of large firms using industry returns
large_by_ind = calculate_stuff(large, 'INDUSTRY')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Pridct the returns of large firms using fundamental returns
large_by_fund = calculate_stuff(large, 'TNIC_RET')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Predict the returns of large firms by the difference
large_by_diff = calculate_stuff(large, 'ind-fund')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Delete dataframes that have done their jobs
del small
del large


# What is today's date
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\cap_analysis\\'
else:
    location = path + '\\Data\\' + folder + '\\'
    
# Name sheet on whether after 2009
if folder is None:
    if post_2009 == True:  
        writer = pd.ExcelWriter(location + 'post_2009_cap_factor_loadings_monthly_cut_' + str(cut) +  '_' + today + '.xlsx')
    else:    
        writer = pd.ExcelWriter(location + 'cap_factor_loadings_monthly_cut_' + str(cut) +  '_' + today + '.xlsx')
else:
    if post_2009 == True:  
        writer = pd.ExcelWriter(location + 'Other//post_2009_cap_factor_loadings_monthly_cut_' + str(cut) +  '_' + today + '.xlsx')
    else:    
        writer = pd.ExcelWriter(location + 'Other//cap_factor_loadings_monthly_cut_' + str(cut) +  '_' + today + '.xlsx')
    
ind_fund_quants = ['Q1', 'Q5', 'Q5-Q1']
diff_quants = ['Q1', 'Q5', 'Q1-Q5']

temp = cf.OLS_results(small_by_ind, FF, ind_fund_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SI')
temp[1].to_excel(writer, sheet_name = 'R^2 SI')
cf.process_rtn(small_by_ind, ind_fund_quants).to_excel(writer, sheet_name = 'Return stats SI')


temp = cf.OLS_results(small_by_fund, FF, ind_fund_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SF')
temp[1].to_excel(writer, sheet_name = 'R^2 SF')
cf.process_rtn(small_by_fund, ind_fund_quants).to_excel(writer, sheet_name = 'Return stats SF')

temp = cf.OLS_results(small_by_diff, FF, diff_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SD')
temp[1].to_excel(writer, sheet_name = 'R^2 SD')
cf.process_rtn(small_by_diff, diff_quants).to_excel(writer, sheet_name = 'Return stats SD')

temp = cf.OLS_results(large_by_ind, FF, ind_fund_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LI')
temp[1].to_excel(writer, sheet_name = 'R^2 LI')
cf.process_rtn(large_by_ind, ind_fund_quants).to_excel(writer, sheet_name = 'Return stats LI')

temp = cf.OLS_results(large_by_fund, FF, ind_fund_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LF')
temp[1].to_excel(writer, sheet_name = 'R^2 LF')
cf.process_rtn(large_by_fund, ind_fund_quants).to_excel(writer, sheet_name = 'Return stats LF')

temp = cf.OLS_results(large_by_diff, FF, diff_quants)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LD')
temp[1].to_excel(writer, sheet_name = 'R^2 LD')
cf.process_rtn(large_by_diff, diff_quants).to_excel(writer, sheet_name = 'Return stats LD')

writer.save()
        
# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('Predict small firm by industry')

axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('Predict small firm by fundamentals')

axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 0].legend(loc = "upper left")
axs[2, 0].set_title('Predict small firm by industry minus fundamentals')

axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('Predict large firm by industry')

axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('Predict large firm by fundamentals')

axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 1].legend(loc = "upper left")
axs[2, 1].set_title('Predict large firm by industry minus fundamentals')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)
if post_2009 == True:   
    fig.savefig(location + 'Charts\\post_2009_cap_returns_monthly_' + str(cut) +  '_' + today + '.png')
else:
    fig.savefig(location + 'Charts\\cap_returns_monthly_' + str(cut) +  '_' + today + '.png')
plt.show()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), ' minutes total')

