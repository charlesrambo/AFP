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

# Load weekly return data
fundamentals = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")

# Load needed columns of monthly return data
columns = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY'])

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN'], axis = 0, how = 'any', inplace = True)
columns.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert dates to datetime objects
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")
columns['DATE'] = pd.to_datetime(columns['DATE'], format = "%Y-%m-%d")

# Get year and month for weekly data
fundamentals['year'] = fundamentals['DATE'].dt.year

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = columns['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
columns['year'] = faux_date.dt.year

# Drop DATE column
columns.drop('DATE', axis = 1, inplace = True)

# Merge on year, month, and DW Instrument
fundamentals = fundamentals.merge(columns, on = ['year', 'DW_INSTRUMENT_ID'])

del columns
del faux_date

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Drop values missing instrument ID or gvkey
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY', 'RETURN'], axis = 0, how = 'any', inplace = True)

# Rename MAIN_KEY to gvkey1 to avoid confusion
fundamentals.rename(columns = {'MAIN_KEY':'gvkey1'}, inplace = True)

# Convert gvkey1 to integer
fundamentals['gvkey1'] = fundamentals['gvkey1'].astype('int64')

# Load Hoberge data
hoberg = pd.read_csv(path + "\\Data\\tnic2_data.txt", sep = '\t')

# Add one to year to map score onto previous and not current year
hoberg['year'] = 1 + hoberg['year']

# Cut Hoberge data by score
hoberg = hoberg[hoberg.score > cut]

# Drop score column
hoberg.drop('score', axis = 1, inplace = True)

# Get columns for right side of merge
rtn = fundamentals[['DATE', 'gvkey1', 'DW_INSTRUMENT_ID', 'RETURN']]

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

# Compute sum of returns
fundamentals['TNIC_RET'] = fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID'])['RETURN2'].transform('mean')

# Record number of firms in fundemental group
fundamentals['N'] = fundamentals.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')

# Keep only useful columns
fundamentals = fundamentals[['DATE', 'DW_INSTRUMENT_ID', 'gvkey1', 'TNIC_RET', 'N']]

# Rename columns to avoid confusion
fundamentals = fundamentals.rename(columns = {'gvkey1':'MAIN_KEY'})

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Convert to pandas
fundamentals.to_csv(path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv', index = False)

print(0.5 * int((time.time() - start_time)/30), ' minutes total')