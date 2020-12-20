import pandas as pd
import time

# Record start time
start_time = time.time()

# What is the cut
cut = 0.10

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load fuller return data
fundamentals = pd.read_csv(path + "\\Data\\univIMIUSMonthlyReturns.csv")

# Load stock data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY'])

# Load clusters
clusters = pd.read_csv(path + "\\Data\\Clusters\\clusters_cut_0.1_linkage_complete.csv")

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)
stocks_month.drop_duplicates(inplace = True)

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN'], axis = 0, how = 'any', inplace = True)
stocks_month.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert to datetime object
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Obtain year
fundamentals['year'] = fundamentals['DATE'].dt.year

# This could be forward looking
stocks_month['year'] = stocks_month['DATE'].dt.year

# Drop date columns from stocks_month
stocks_month.drop('DATE', axis = 1, inplace = True) 

# Merge fundamentals with stocks_month; forward looking problem
fundamentals = fundamentals.merge(stocks_month, on = ['year', 'DW_INSTRUMENT_ID'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Delete stocks_month from memory
del stocks_month

# Add 1 to the year of clusters
clusters['year'] = 1 + clusters['year']

# Merge clusters and fundamentals
fundamentals = fundamentals.merge(clusters, on = ['MAIN_KEY', 'year'])

del clusters

# How much time has it been so far?
print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

# Get columns for right side of merge
rtn = fundamentals[['DATE', 'MAIN_KEY', 'DW_INSTRUMENT_ID', 'RETURN', 'cluster']]

# Rename columns
rtn.rename(columns = {'MAIN_KEY':'MAIN_KEY2', 'DW_INSTRUMENT_ID':'DW_INSTRUMENT_ID2', 'RETURN':'RETURN2'}, inplace = True)

# Drop year column
fundamentals.drop('year', axis = 1, inplace = True)

# Merge fundementals and return data
fundamentals = fundamentals.merge(rtn, on = ['DATE','cluster'])

# Remove return data from memory
del rtn

# Drop observations where MAIN_KEY == MAIN_KEY2
fundamentals = fundamentals.loc[fundamentals['MAIN_KEY'] != fundamentals['MAIN_KEY2'], :]

# Compute mean of returns for equal weighting
fundamentals['TNIC_RET'] = fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID'])['RETURN2'].transform('mean')

# Record number of firms in fundemental group
fundamentals['N'] = fundamentals.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')

# Keep only useful columns
fundamentals = fundamentals[['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'TNIC_RET', 'N']]

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Convert to pandas
fundamentals.to_csv(path + '\\Data\\fundamental_mods\\monthly_fund_rtn_cluster_cut_' + str(cut) + '.csv', index = False)

print(0.5 * int((time.time() - start_time)/30), ' minutes total')
