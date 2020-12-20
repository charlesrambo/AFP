import pandas as pd
import time

# Record start time
start_time = time.time()

# What is the cut
cut = 0.10

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load stock data
fundamentals = pd.read_csv(path + "\\Data\\univIMIUSMonthlyReturns.csv")

stocks = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'TOTAL_SHARES', 'PRICE_UNADJUSTED'])

# Convert to datetime object
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")
stocks['DATE'] = pd.to_datetime(stocks['DATE'], format = "%Y-%m-%d")

# Obtain year
fundamentals['year'] = fundamentals['DATE'].dt.year
stocks['year'] =stocks['DATE'].dt.year

# Merge on year
fundamentals = fundamentals.merge(stocks[['year', 'DW_INSTRUMENT_ID', 'MAIN_KEY']], on = ['year', 'DW_INSTRUMENT_ID'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Forward fill TOTAL_SHARES and PRICE_UNADJUSTED
stocks['TOTAL_SHARES'] = stocks.groupby('DW_INSTRUMENT_ID')['TOTAL_SHARES'].ffill()
stocks['PRICE_UNADJUSTED'] = stocks.groupby('DW_INSTRUMENT_ID')['PRICE_UNADJUSTED'].ffill()

# Create market equity column
stocks['ME'] = stocks['TOTAL_SHARES'] * stocks['PRICE_UNADJUSTED']

# Sort values
stocks.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Shift results; some dates may be more than a month previous
stocks['ME_lag'] = stocks.groupby('DW_INSTRUMENT_ID')['ME'].shift(1)

# Drop unneeded columns
stocks.drop(['TOTAL_SHARES', 'PRICE_UNADJUSTED', 'ME', 'TOTAL_SHARES', 'PRICE_UNADJUSTED', 'MAIN_KEY'], axis = 1, inplace = True)

# Merge fundamentals and stocks on
fundamentals = fundamentals.merge(stocks, on = ['DATE', 'DW_INSTRUMENT_ID', 'year'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Record 25% quantile for each month
fundamentals['25'] = stocks[['DATE', 'ME_lag']].groupby('DATE')['ME_lag'].quantile(0.25)

# Replace missing ME_lags with 25-percentile value
fundamentals.loc[fundamentals['ME_lag'].isna(), 'ME_lag'] = fundamentals.loc[fundamentals['ME_lag'].isna(), '25']

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN', 'ME_lag'], axis = 0, how = 'any', inplace = True)

# Drop 25
fundamentals.drop('25', axis = 1, inplace = True)

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
fundamentals.to_csv(path + '\\Data\\fundamental_mods\\monthly_fund_rtn_val_wt_cut_' + str(cut) + '.csv', index = False)

print(0.5 * int((time.time() - start_time)/30), ' minutes total')