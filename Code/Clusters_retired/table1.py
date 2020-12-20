import numpy as np
import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd, Week
import time

# Record start time
start_time = time.time()
 
# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What type of linkage?
linkage = 'complete'

# What is the cut?
cut = 0.15

# Load clusters 
clusters = pd.read_csv(path + '\\Data\\Clusters\\clusters_cut_'+ str(cut) + '_linkage_' + linkage + '.csv')

# Hoberg score in year t is matched to stock data in year t + 1
clusters['year'] = 1 + clusters['year']

# Load return data
stocks_week = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Convert to CRSP and Compustat dates to date time objects
stocks_week['DATE'] = pd.to_datetime(stocks_week['DATE'], format = "%Y-%m-%d") 
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Introduce year variable
stocks_week['year'] = stocks_week['DATE'].dt.year

# Introduce month variable
stocks_week['month'] = stocks_week['DATE'].dt.month

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
stocks_month['year'] = faux_date.dt.year
stocks_month['month'] = faux_date.dt.month

del faux_date

# Select columns for future merge
columns = stocks_month.loc[:, ['year', 'month', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'SEC_KEY', 'INDUSTRY', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]

# Drop unneeded columns from stocks_month
stocks_month = stocks_month.loc[:, ['DATE', 'year', 'DW_INSTRUMENT_ID', 'TOTAL_SHARES', 'MAIN_KEY', 'INDUSTRY', 'RETURN', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]

# Fix month and year columns for monthly data
stocks_month['year'] = stocks_month['DATE'].dt.year
stocks_month['month'] = stocks_month['DATE'].dt.month

# Merge with weekly data
stocks_week = stocks_week.merge(columns, on = ['year', 'month', 'DW_INSTRUMENT_ID'])

# Merge with clusters
stocks_week = stocks_week.merge(clusters, on = ['year', 'MAIN_KEY'])
stocks_month = stocks_month.merge(clusters, on = ['year', 'MAIN_KEY'])

# Only consider observations before the year 2010
stocks_week = stocks_week.loc[stocks_week['year'] < 2010, :]
stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]

# Drop year and month
stocks_week.drop(['year', 'month'], axis = 1, inplace = True)
stocks_month.drop('year', axis = 1, inplace = True)

# Convert returns to decimals
stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']

# Calculate fundemental return; equal weighted within cluster
stocks_week['fund'] = stocks_week.groupby(['DATE', 'cluster'])['RETURN'].transform('mean')
stocks_month['fund'] = stocks_month.groupby(['DATE', 'cluster'])['RETURN'].transform('mean')

# Calculate faux SIC return; equal weighted within each faux SIC
stocks_week['ind'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')
stocks_month['ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Calculate firm return minus industry return 
stocks_week['r-ind'] = stocks_week['RETURN'] - stocks_week['ind']
stocks_month['r-ind'] = stocks_month['RETURN'] - stocks_month['ind']

# Calculate official minus fundemental
stocks_week['ind-fund'] = stocks_week['ind'] - stocks_week['fund']
stocks_month['ind-fund'] = stocks_month['ind'] - stocks_month['fund']

# Construct r - fund
stocks_week['r-fund'] = stocks_week['RETURN'] - stocks_week['fund']
stocks_month['r-fund'] = stocks_month['RETURN'] - stocks_month['fund']

# Create SIC and cluster counts
stocks_week['INDUSTRY_n'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
stocks_week['cluster_n'] = stocks_week.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')
stocks_month['INDUSTRY_n'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
stocks_month['cluster_n'] = stocks_month.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')

# Define function to calculate cumulative returns
prod = lambda x: (1 + x).prod() - 1

# Sort values
stocks_month.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Calculate cumulative returns
stocks_month['cr'] = stocks_month.groupby('DW_INSTRUMENT_ID')['RETURN'].transform(lambda x: x.shift(2).rolling(11).apply(prod))

# Calculate total market equity
stocks_month['TOTAL_MCAP'] = stocks_month['PRICE_UNADJUSTED'] * stocks_month['TOTAL_SHARES']

# Get equity information; use month of June
good_firm = (stocks_month['INDUSTRY_n'] > 4) & (stocks_month['cluster_n'] > 4) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['DATE'].dt.month == 6) & (stocks_month['TOT_EQUITY'] > 0)
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

# Trime at the 99th percentile
equity.loc[equity['ln_BMR'] > equity['ln_BMR'].quantile(0.99),'ln_BMR'] = equity['ln_BMR'].quantile(0.99)

# Only consider firms with more than 5 completing firms based both on inustry and cluster, positive book equity
good_firm = (stocks_week['INDUSTRY_n'] > 4) & (stocks_week['cluster_n'] > 4) & (stocks_week['PRICE_UNADJUSTED'] >= 5) & (stocks_week['TOT_EQUITY'] > 0)
stocks_week = stocks_week.loc[good_firm, :]

good_firm = (stocks_month['INDUSTRY_n'] > 4) & (stocks_month['cluster_n'] > 4) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['TOT_EQUITY'] > 0)
stocks_month = stocks_month.loc[good_firm, :]

# Save results
today = str(pd.to_datetime("today"))[0:10]
writer = pd.ExcelWriter(path + '\\Data\\clusters_table1_results\\table1_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

# Yearly
stocks_month['INDUSTRY_n'].describe().to_excel(writer, sheet_name = 'Industry')

stocks_month['cluster_n'].describe().to_excel(writer, sheet_name = 'Hoberg')

equity['ln_BMR'].describe().to_excel(writer, sheet_name = 'June Book-to-Market ratio')

# Monthly
stocks_month['RETURN'].describe().to_excel(writer, sheet_name = 'Monthly rtn')

stocks_month['ind'].describe().to_excel(writer, sheet_name = 'Monthly offi rtn')

stocks_month['fund'].describe().to_excel(writer, sheet_name = 'Monthly fund rtn')

stocks_month['ind-fund'].describe().to_excel(writer, sheet_name = 'Monthly offi minus fund rtn')

stocks_month['cr'].describe().to_excel(writer, sheet_name = 'Cum rtn t-12 to t-2')

stocks_month['ln_MCAP_lag'].describe().to_excel(writer, sheet_name = 'Monthly log of ME')

# Weekly
stocks_week['RETURN'].describe().to_excel(writer, sheet_name = 'Weekly rtn')

stocks_week['ind'].describe().to_excel(writer, sheet_name = 'Weekly offi rtn')

stocks_week['fund'].describe().to_excel(writer, sheet_name = 'Weekly fund rtn')

stocks_week['ind-fund'].describe().to_excel(writer, sheet_name = 'Weekly offi minus fund rtn')

writer.save()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')
