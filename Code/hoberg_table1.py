import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd
import time
import custom_functions as cf

# Record start time
start_time = time.time()
 
# Record path
path = r'C:\\Users\\rambocha\\Desktop\\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009
post_2009 = True

# Where are the fundamental returns
file1 = path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'
file2 = path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = None

# Get the month return data
stocks_month = cf.prep_monthly_rtn(path, cut, file1, post_2009)

# Get the weekly return data
stocks_week = cf.prep_weekly_rtn(path, cut, file2, post_2009)

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Calculate faux SIC return; equal weighted within each faux SIC
stocks_week['ind'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')
stocks_month['ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Winsorize functions
stocks_week = cf.winsorize(stocks_week, 'ind')
stocks_month = cf.winsorize(stocks_month, 'ind')

# Calculate firm return minus industry return 
stocks_week['r-ind'] = stocks_week['RETURN'] - stocks_week['ind']
stocks_month['r-ind'] = stocks_month['RETURN'] - stocks_month['ind']

# Calculate official minus fundemental
stocks_week['ind-fund'] = stocks_week['ind'] - stocks_week['TNIC_RET']
stocks_month['ind-fund'] = stocks_month['ind'] - stocks_month['TNIC_RET']

# Construct r - fund
stocks_week['r-fund'] = stocks_week['RETURN'] - stocks_week['TNIC_RET']
stocks_month['r-fund'] = stocks_month['RETURN'] - stocks_month['TNIC_RET']

# Create SIC counts
stocks_week['INDUSTRY_n'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
stocks_month['INDUSTRY_n'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Define function to calculate cumulative returns
prod = lambda x: (1 + x).prod() - 1

# Sort values
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Calculate cumulative returns
stocks_month['cr'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].shift(2).rolling(11).apply(prod)

# Calculate total market equity
stocks_month['TOTAL_MCAP'] = stocks_month['PRICE_UNADJUSTED'] * stocks_month['TOTAL_SHARES']

# Get equity information; use month of June
good_firm = (stocks_month['INDUSTRY_n'] > 2) & (stocks_month['N'] > 2) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['DATE'].dt.month == 6) & (stocks_month['TOT_EQUITY'] > 0)
equity = stocks_month.loc[good_firm, ['TOTAL_MCAP', 'TOT_EQUITY']]

# Calculate natural log of market equity
stocks_month['ln_MCAP'] = np.log(stocks_month['TOTAL_MCAP'])

# Calculate natural log of book to market ratio
equity['ln_BMR'] = np.log(equity['TOT_EQUITY']/equity['TOTAL_MCAP'])

# Sort values
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Shift log of market cap
stocks_month['ln_MCAP_lag'] = stocks_month.groupby('DW_INSTRUMENT_ID')['ln_MCAP'].shift(1)

# Check if valid
stocks_month['valid'] = stocks_month['DATE'].shift(1) + dt.timedelta(days = 7) + MonthEnd(0) == stocks_month['DATE'] + MonthEnd(0)

# Remove invalid observations
stocks_month.loc[stocks_month['valid'] == False, 'ln_MCAP_lag'] = np.nan

# Trim at the 99th percentile
equity.loc[equity['ln_BMR'] > equity['ln_BMR'].quantile(0.99),'ln_BMR'] = equity['ln_BMR'].quantile(0.99)

# Only consider firms with more than 5 completing firms based both on inustry and Hoberg grouping, positive book equity
good_firm = (stocks_week['INDUSTRY_n'] > 2) & (stocks_week['N'] > 2) & (stocks_week['PRICE_UNADJUSTED'] >= 5) & (stocks_week['TOT_EQUITY'] > 0)
stocks_week = stocks_week.loc[good_firm, :]

good_firm = (stocks_month['INDUSTRY_n'] > 2) & (stocks_month['N'] > 2) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['TOT_EQUITY'] > 0)
stocks_month = stocks_month.loc[good_firm, :]

# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\hoberg_table1_results\\'
else:
    location = path + '\\Data\\' + folder + '\\'

# Name sheet on whether after 2009
if post_2009 == True:  
    writer = pd.ExcelWriter(location + 'post_2009_table_1_cut_' + str(cut) + '_' + today + '.xlsx')
else:    
    writer = pd.ExcelWriter(location + 'table_1_cut_' + str(cut) + '_' + today + '.xlsx')

# Yearly
stocks_month['INDUSTRY_n'].describe().to_excel(writer, sheet_name = 'Industry')

stocks_month['N'].describe().to_excel(writer, sheet_name = 'Hoberg')

equity['ln_BMR'].describe().to_excel(writer, sheet_name = 'June Book-to-Market ratio')

# Monthly
stocks_month['RETURN'].describe().to_excel(writer, sheet_name = 'Monthly rtn')

stocks_month['ind'].describe().to_excel(writer, sheet_name = 'Monthly offi rtn')

stocks_month['TNIC_RET'].describe().to_excel(writer, sheet_name = 'Monthly fund rtn')

stocks_month['ind-fund'].describe().to_excel(writer, sheet_name = 'Monthly offi minus fund rtn')

stocks_month['cr'].describe().to_excel(writer, sheet_name = 'Cum rtn t-12 to t-2')

stocks_month['ln_MCAP_lag'].describe().to_excel(writer, sheet_name = 'Monthly log of ME')

# Weekly
stocks_week['RETURN'].describe().to_excel(writer, sheet_name = 'Weekly rtn')

stocks_week['ind'].describe().to_excel(writer, sheet_name = 'Weekly offi rtn')

stocks_week['TNIC_RET'].describe().to_excel(writer, sheet_name = 'Weekly fund rtn')

stocks_week['ind-fund'].describe().to_excel(writer, sheet_name = 'Weekly offi minus fund rtn')

writer.save()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes')
