import pandas as pd
import numpy as np

# What is the cut?
cut = 0.07

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load return data
fundamentals = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'RETURN'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)

# Drop values missing instrument ID or gvkey
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY', 'RETURN'], axis = 0, how = 'any', inplace = True)

# Convert to datetime object
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")

# Only consider observations on 1/31/2000
fundamentals = fundamentals.loc[fundamentals['DATE'] == pd.to_datetime('1/31/2000', format = '%m/%d/%Y'), :]

# Obtain year
fundamentals['year'] = fundamentals['DATE'].dt.year

# Rename MAIN_KEY to gvkey1 to avoid confusion
fundamentals.rename(columns = {'MAIN_KEY':'gvkey1'}, inplace = True)

# Convert gvkey1 to integer
fundamentals['gvkey1'] = fundamentals['gvkey1'].astype('int64')

# Record gvkeys
gvkeys = np.unique(fundamentals['gvkey1'])

# Load Hoberge data
hoberg = pd.read_csv(path + "\\Data\\tnic2_data.txt", sep = '\t')

# Add one to year to map score onto previous and not current year
hoberg['year'] = 1 + hoberg['year']

# Cut Hoberge data by score
hoberg = hoberg.loc[hoberg.score > cut]

# Only select gvkeys contained in fundementals
hoberg = hoberg.loc[hoberg.isin({'gvkey2':gvkeys}).gvkey2]

# Delete gvkeys
del gvkeys

# Calculate sum returns at a given date of firms with the same gvkey
fundamentals['RETURN_sum'] = fundamentals.groupby(['DATE', 'gvkey1'])['RETURN'].transform('sum')

# Calculate number of firms in each gvkey
fundamentals['N'] = fundamentals.groupby(['DATE', 'gvkey1'])['DW_INSTRUMENT_ID'].transform('nunique')

# Select columns for right side of merge
rtn = fundamentals[['DATE', 'RETURN', 'DW_INSTRUMENT_ID', 'gvkey1']]

# Rename gvkey1 in rtn
rtn.rename(columns = {'gvkey1':'gvkey2'}, inplace = True)

# Drop Return column
fundamentals.drop('RETURN', axis = 1, inplace = True)

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
fundamentals['TNIC_RET'] = fundamentals['RETURN_sum'] + fundamentals.groupby(['DATE', 'DW_INSTRUMENT_ID_x'])['RETURN'].transform('sum')

# Drop the return on the original dataframe and RETURN_sum
fundamentals.drop('RETURN_sum', axis = 1, inplace = True)

# Record number of firms in fundemental group
fundamentals['N'] = fundamentals['N'] + fundamentals.groupby(['DATE','DW_INSTRUMENT_ID_x'])['DW_INSTRUMENT_ID_y'].transform('nunique')

# Divide by number of firms to obtain equal weighted returns
fundamentals['TNIC_RET'] = fundamentals['TNIC_RET']/fundamentals['N']

# Rename columns so they're the same as in example
fundamentals.rename(columns = {'DW_INSTRUMENT_ID_x':'DW_INSTRUMENT_ID', 'DW_INSTRUMENT_ID_y':'DW_INSTRUMENT_ID2', 'score':'SCORE', 'gvkey1':'MAIN_KEY', 'gvkey2':'MAIN_KEY2'}, inplace = True)

# Subset fundamentals
fundamentals = fundamentals[['DATE', 'DW_INSTRUMENT_ID', 'DW_INSTRUMENT_ID2', 'MAIN_KEY', 'MAIN_KEY2', 'SCORE', 'RETURN', 'TNIC_RET', 'N']]

# Drop duplicates
fundamentals = fundamentals.drop_duplicates()

# Convert to cvs
fundamentals.to_csv(path + '\\Data\\request_cut_' + str(cut) + '.csv', index = False)
