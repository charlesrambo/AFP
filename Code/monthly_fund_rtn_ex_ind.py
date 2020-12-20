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
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)
stocks_month.drop_duplicates(inplace = True)

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN'], axis = 0, how = 'any', inplace = True)
stocks_month.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY'], axis = 0, how = 'any', inplace = True)

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
rtn = fundamentals[['DATE', 'gvkey1', 'DW_INSTRUMENT_ID', 'RETURN', 'INDUSTRY']]

# Rename columns
rtn.rename(columns = {'gvkey1':'gvkey2', 'DW_INSTRUMENT_ID':'DW_INSTRUMENT_ID2', 'RETURN':'RETURN2', 'INDUSTRY':'INDUSTRY2'}, inplace = True)

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

# Exclude firms of same industry
fundamentals = fundamentals.loc[fundamentals['INDUSTRY'] != fundamentals['INDUSTRY2'], :]

# Drop INDUSTY and Industry2
fundamentals.drop(['INDUSTRY', 'INDUSTRY2'], axis = 1, inplace = True)

# Compute mean of returns for equal weighting
fundamentals['TNIC_RET'] = fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID'])['RETURN2'].transform('mean')

# Record number of firms in fundemental group
fundamentals['N'] = fundamentals.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')

# Keep only useful columns
fundamentals = fundamentals[['DATE', 'DW_INSTRUMENT_ID', 'gvkey1', 'TNIC_RET', 'N']]

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Rename columns to avoid confusion
fundamentals.rename(columns = {'gvkey1':'MAIN_KEY'}, inplace = True)

# Convert to pandas
fundamentals.to_csv(path + '\\Data\\fundamental_mods\\monthly_fund_rtn_ex_ind_cut_' + str(cut) + '.csv', index = False)

print(0.5 * int((time.time() - start_time)/30), ' minutes total')
