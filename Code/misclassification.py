import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# Record start time
start_time = time.time()

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load fuller return data
fundamentals = pd.read_csv(path + "\\Data\\univIMIUSMonthlyReturns.csv")

# Define cut for average industry misclassification within sector
sector_cut = 0.10

# Load stock data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'SECTOR',
'INDUSTRY', 'NAME'])

# Drop duplicates
fundamentals.drop_duplicates(inplace = True)
stocks_month.drop_duplicates(inplace = True)

# Drop values with missing instrument ID, gvkey, or return
fundamentals.dropna(subset = ['DW_INSTRUMENT_ID', 'RETURN'], axis = 0, how = 'any', inplace = True)
stocks_month.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Drop return column
fundamentals.drop('RETURN', axis = 1, inplace = True)

# Convert to datetime object
fundamentals['DATE'] = pd.to_datetime(fundamentals['DATE'], format = "%Y-%m-%d")
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Only consider observations in December
fundamentals = fundamentals.loc[fundamentals['DATE'].dt.month == 12, :]
stocks_month = stocks_month.loc[stocks_month['DATE'].dt.month == 12, :]

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

# Get columns for right side of merge
rtn = fundamentals[['DATE', 'gvkey1', 'DW_INSTRUMENT_ID', 'INDUSTRY', 'NAME']]

# Rename columns
rtn.rename(columns = {'gvkey1':'gvkey2', 'DW_INSTRUMENT_ID':'DW_INSTRUMENT_ID2', 'INDUSTRY':'INDUSTRY2', 'NAME':'NAME2'}, inplace = True)

# How much time has it been so far?
print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

# Load Hoberge data
hoberg = pd.read_csv(path + "\\Data\\tnic2_data.txt", sep = '\t')

# Add one to year to map score onto previous and not current year
hoberg['year'] = 1 + hoberg['year']

# Initialize data frame for fraction misclassified
misclass = pd.DataFrame()

# Save the date
today = str(pd.to_datetime("today"))[0:10]

# Define function to shorten processing code
def create_data(cut):
    
    # Cut Hoberge data by score
    hoberg_cut = hoberg.loc[hoberg.score > cut]

    # Drop score column
    hoberg_cut.drop('score', axis = 1, inplace = True)

    # Merge fundamentals and Hoberg
    fund_merg = fundamentals.merge(hoberg_cut, on = ['gvkey1', 'year'])

    # Delete Hoberg data
    del hoberg_cut

    # Merge fundementals and return data
    fund_merg = fund_merg.merge(rtn, on = ['DATE','gvkey2'])
    
    # Drop duplicates
    fund_merg.drop_duplicates(inplace = True)
    
    # Get boolean if INDUSTRY != INDUSTRY2
    fund_merg['misclass_temp'] = fund_merg['INDUSTRY'] != fund_merg['INDUSTRY2']
    
    # Get total missclassifications in each group
    fund_merg['misclass_N'] = fund_merg.groupby(['DATE','DW_INSTRUMENT_ID'])['misclass_temp'].transform('sum')

    # Record number of firms in fundemental group
    fund_merg['N'] = fund_merg.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')
    
    # Get fraction misclassified
    fund_merg['misclass_frac'] = fund_merg['misclass_N']/fund_merg['N']
    
    # Drop misclass_temp
    fund_merg.drop('misclass_temp', axis = 1, inplace = True)
    
    return fund_merg
    

# Define list of cuts
cuts = [0, 0.05, 0.10, 0.15]

for cut in cuts:

    # Do calculations for particular cut
    fund_merg = create_data(cut)
                 
    # Convert to pandas
    fund_merg.loc[fund_merg['year'].isin([1998, 2003, 2008, 2013, 2018]), ['DATE', 'NAME', 'DW_INSTRUMENT_ID', 'INDUSTRY', 'NAME2', 'DW_INSTRUMENT_ID2', 'INDUSTRY2', 'misclass_frac', 'N']].to_csv(path + '\\Data\\industry\\hoberg_groups_' + str(cut) + '_' + today + '.csv', index = False)

    # Get mean misclassification for each date
    misclass['misclass_frac_cut_' + str(cut)] = fund_merg.groupby('DATE')['misclass_frac'].mean()
    
    print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

# Reset index for misclass
misclass.reset_index(inplace = True)

# Calculate the average by sector

# Initialize data frame for fraction misclassified
sectors = pd.DataFrame()

# Perform data Engineering
fund_merg = create_data(sector_cut)

# Drop observations where sector is missing
fund_merg.dropna(subset = ['SECTOR'], inplace = True)

# Get mean misclassification for each date and sector combination
sectors['misclass_frac'] = fund_merg.groupby(['DATE', 'SECTOR'])['misclass_frac'].mean()
    
# Reset index
sectors.reset_index(inplace = True)

# Get a list of sectors befor pivot
list_of_sectors = np.unique(sectors['SECTOR'])

sectors = sectors.pivot(index = 'DATE', columns = 'SECTOR', values = 'misclass_frac').reset_index()

means = sectors.mean(axis = 0)
stand_dev = sectors.std(axis = 0)

result = pd.concat([means, stand_dev], axis = 1, ignore_index = True)
result.rename(columns = {0:'mean', 1:'std'}, inplace = True)

writer = pd.ExcelWriter(path + '//Data//industry//missclass_by_sector_cut_' + str(cut) + '_' + today + '.xlsx')

result.to_excel(writer, sheet_name = 'mean and std by sector')

writer.save()

del means, stand_dev, result

fig, axs = plt.subplots(2, 1)

n = len(list_of_sectors)
i = 1

for sector in list_of_sectors:
    
    if i <= int(n/2):
        axs[0].plot(sectors['DATE'], sectors[sector], label = sector)
        axs[0].set_xlabel('DATE')
        axs[0].set_ylabel('Fraction') 
        axs[0].legend(loc = 'upper left')
        axs[0].set_title('Misclassification')
    else:
        axs[1].plot(sectors['DATE'], sectors[sector], label = sector)
        axs[1].set_xlabel('DATE')
        axs[1].set_ylabel('Fraction') 
        axs[1].legend(loc = 'upper left')
        axs[1].set_title('Misclassification')
       
    i+=1
    
fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)

fig.savefig(path + '//Data//industry//Charts//misclass_frac_by_sector_cut_0.1_' + today + '.png')

plt.show()


fig, ax = plt.subplots(1, 1)

for cut in cuts:
    ax.plot(misclass['DATE'], misclass['misclass_frac_cut_' + str(cut)], label = str(cut))
   
ax.set_xlabel('DATE')
ax.set_ylabel('Fraction') 
ax.legend(loc = 'upper left')
ax.set_title('Misclassification')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)

fig.savefig(path + '//Data//industry//Charts//misclass_frac_' + str(cut) + '_' + today + '.png')

plt.show()

print(0.5 * int((time.time() - start_time)/30), ' minutes total')
