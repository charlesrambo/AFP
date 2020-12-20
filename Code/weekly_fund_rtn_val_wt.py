import pandas as pd
import datetime as dt
from pandas.tseries.offsets import MonthEnd, Week
import time

# Record start time
start_time = time.time()

# What is the cut
cut = 0.10

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load stock data
fundamentals = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")

# Load needed columns of monthly return data
columns = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'TOTAL_SHARES', 'PRICE_UNADJUSTED'])

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN'], axis = 0, how = 'any', inplace = True)
columns.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert to datetime object
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")
columns['DATE'] = pd.to_datetime(columns['DATE'], format = "%Y-%m-%d")

# Forward fill TOTAL_SHARES and PRICE_UNADJUSTED
columns['TOTAL_SHARES'] = columns.groupby('DW_INSTRUMENT_ID')['TOTAL_SHARES'].ffill()
columns['PRICE_UNADJUSTED'] = columns.groupby('DW_INSTRUMENT_ID')['PRICE_UNADJUSTED'].ffill()

# Calculate ME_lag; not lagged yet however
columns['ME_lag'] = columns['TOTAL_SHARES'] * columns['PRICE_UNADJUSTED']

# Create ME dataframe
ME = columns[['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'ME_lag']]

# Drop unneeded columns
columns.drop(['TOTAL_SHARES', 'PRICE_UNADJUSTED', 'ME_lag'], axis = 1, inplace = True)

# Obtain year
fundamentals['year'] = fundamentals['DATE'].dt.year
fundamentals['month'] = fundamentals['DATE'].dt.month

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = columns['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
columns['year'] = faux_date.dt.year

# Create year and month columns for ME
ME['year'] = faux_date.dt.year
ME['month'] = faux_date.dt.month

# Drop columns
columns.drop('DATE', axis = 1, inplace = True)
ME.drop('DATE', axis = 1, inplace = True)

# Drop values with missing instrument ID, gvkey, or return
ME.dropna(subset = ['ME_lag'], axis = 0, how = 'any', inplace = True)

# Merge columns with fundamentals
fundamentals = fundamentals.merge(columns, on = ['year', 'DW_INSTRUMENT_ID'])

# Merge ME and fundamentals
fundamentals = fundamentals.merge(ME, on = ['year', 'month', 'MAIN_KEY', 'DW_INSTRUMENT_ID'], how = 'left')

del columns
del ME

# Get ME_lag of the 25th percentile each week
fundamentals['25'] = fundamentals[['DATE', 'ME_lag']].groupby('DATE')['ME_lag'].quantile(0.25)

# Replace missing ME_lags with 25-percentile value
fundamentals.loc[fundamentals['ME_lag'].isna(), 'ME_lag'] = fundamentals.loc[fundamentals['ME_lag'].isna(), '25']

# drop month column
fundamentals.drop(['month', '25'], axis = 1, inplace = True)

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Rename MAIN_KEY to gvkey1 to avoid confusion
fundamentals.rename(columns = {'MAIN_KEY':'gvkey1'}, inplace = True)

# Convert gvkey1 to integer
fundamentals['gvkey1'] = fundamentals['gvkey1'].astype('int64')

# How much time has it been so far?
print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

# Load Hoberge data
hoberg = pd.read_csv(path + "\\Data\\tnic2_data.txt", sep = '\t')

# Add one to year to map score onto previous and not current year
hoberg['year'] = 1 + hoberg['year']

# Cut Hoberge data by score
hoberg = hoberg.loc[hoberg.score > cut]

# Drop score column
hoberg.drop('score', axis = 1, inplace = True)

# Get columns for right side of merge
rtn = fundamentals[['DATE', 'gvkey1', 'DW_INSTRUMENT_ID', 'RETURN', 'ME_lag']]

# Drop ME_lag from fundamentals
fundamentals.drop('ME_lag', axis = 1, inplace = True)

# Rename columns
rtn.rename(columns = {'gvkey1':'gvkey2', 'DW_INSTRUMENT_ID':'DW_INSTRUMENT_ID2', 'RETURN':'RETURN2'}, inplace = True)

# Merge fundamentals and Hoberg
fundamentals = fundamentals.merge(hoberg, on = ['gvkey1', 'year'])

# Delete Hoberg data
del hoberg

# Drop year column
fundamentals.drop('year', axis = 1, inplace = True)

# Merge fundementals and return data
fundamentals = fundamentals.merge(rtn, on = ['DATE','gvkey2'])

# Remove return data from memory
del rtn

# Collect total ME_lag value
fundamentals['sum'] = fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID'])['ME_lag'].transform('sum')

# Divide ME_lag by sum
fundamentals['ME_lag'] = fundamentals['ME_lag']/fundamentals['sum']

# Multiply RETURN2 by ME_lag
fundamentals['RETURN2'] = fundamentals['RETURN2'] * fundamentals['ME_lag']

# Compute sum of RETURN2 to find vale-weighted returns
fundamentals['TNIC_RET'] = fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID'])['RETURN2'].transform('sum')

# Record number of firms in fundemental group
fundamentals['N'] = fundamentals.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')

# Keep only useful columns
fundamentals = fundamentals[['DATE', 'DW_INSTRUMENT_ID', 'gvkey1', 'TNIC_RET', 'N']]

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Rename columns to avoid confusion
fundamentals.rename(columns = {'gvkey1':'MAIN_KEY'}, inplace = True)

# Convert to pandas
fundamentals.to_csv(path + '\\Data\\fundamental_mods\\weekly_fund_rtn_val_wt_cut_' + str(cut) + '.csv', index = False)

print(0.5 * int((time.time() - start_time)/30), ' minutes total')
