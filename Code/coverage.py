import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import datetime as dt
from pandas.tseries.offsets import MonthEnd
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10
    
# Load hoberg data or alternative
hoberg = pd.read_csv(path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv')

# Import stocks data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'RETURN', 'MAIN_KEY', 'INDUSTRY', 
                                                                            'PRICE_UNADJUSTED', 'TOT_EQUITY', 'TOTAL_SHARES']) 

# Calculate number of firms at each date
stocks_month['firms'] = stocks_month.groupby(['DATE'])['DW_INSTRUMENT_ID'].transform('count')

# Convert DATE column to a date object
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")
hoberg['DATE'] = pd.to_datetime(hoberg['DATE'], format = "%Y-%m-%d")
    
# Create year and month columns
stocks_month['year'] = stocks_month['DATE'].dt.year
        
# Drop missing values
stocks_month.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)
        
# Convert type of MAIN_KEY column
stocks_month['MAIN_KEY'] = stocks_month['MAIN_KEY'].astype('int64')
     
# Merge Hoberg and stocks_month data
stocks_month = stocks_month.merge(hoberg, on = ['DATE', 'MAIN_KEY', 'DW_INSTRUMENT_ID'], how = 'left')
    
del hoberg
   
# Drop MAIN_KEY
stocks_month.drop('MAIN_KEY', axis = 1, inplace = True)
    
# Convert to decimal
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']
stocks_month['TNIC_RET'] = 0.01 * stocks_month['TNIC_RET'] 

# Winsorize columns
stocks_month = cf.winsorize(stocks_month, 'RETURN')

print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

# Define signal and delay
signal = 1 
delay = 0

# Subset to just important columns
stocks_sub = stocks_month[['DATE', 'RETURN', 'TNIC_RET', 'N', 'INDUSTRY', 'DW_INSTRUMENT_ID', 'PRICE_UNADJUSTED', 'TOT_EQUITY', 'firms']]
           
# Compute equal weighted returns
stocks_sub['ind'] = stocks_sub.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')
        
# Compute number of clusters
stocks_sub['ind_N'] = stocks_sub.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
   
# Compute the difference of industry and fundamental returns
stocks_sub['ind-fund'] = stocks_sub['ind'] - stocks_sub['TNIC_RET']
   
# Sort values
stocks_sub.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)
        
# Determine the valid observations
stocks_sub['valid'] = stocks_sub['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == stocks_sub['DATE'] + MonthEnd(0)   
        
# Determine momentum
stocks_sub['MOM'] = stocks_sub['ind-fund'].shift(1 + delay).rolling(signal).apply(cf.prod)
        
# Get rid of the the invalid observations
stocks_sub.loc[stocks_sub['valid'] == False, 'MOM'] = np.nan
        
# Remove observations with undefined momentum or fewer than 5 firms in industry or fundamental group
good_firm = (stocks_sub['TOT_EQUITY'] > 0) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & stocks_sub['MOM'].notna() & (stocks_sub['N'] > 4) & (stocks_sub['ind_N'] > 4)
okay_firms = (stocks_sub['TOT_EQUITY'] > 0) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & stocks_sub['MOM'].notna() 
all_but_eq =  (stocks_sub['PRICE_UNADJUSTED'] >= 5) & stocks_sub['MOM'].notna() & (stocks_sub['N'] > 4) & (stocks_sub['ind_N'] > 4)
all_but_prc = (stocks_sub['TOT_EQUITY'] > 0) & stocks_sub['MOM'].notna() & (stocks_sub['N'] > 4) & (stocks_sub['ind_N'] > 4)
dirty = stocks_sub['MOM'].notna() & stocks_sub['RETURN'].notna()
afp = (stocks_sub['TOT_EQUITY'] > 0) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & stocks_sub['MOM'].notna() & (stocks_sub['N'] > 2) & (stocks_sub['ind_N'] > 2)

clean = stocks_sub.loc[good_firm, :]
all_but_eq = stocks_sub.loc[all_but_eq, :]
all_but_prc = stocks_sub.loc[all_but_prc, :]
unclean = stocks_sub.loc[okay_firms, :]
dirty = stocks_sub.loc[dirty, :]
afp = stocks_sub.loc[afp, :]


# Create function to process clear and dirty
def process_data(data):
    
    # Drop variables that have done their jobs
    data.drop(['valid', 'N', 'ind_N', 'TOT_EQUITY', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)

    # Create quantiles; add 1 to avoid zero-indexing confusion
    data['quintile'] = data[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))

    # Drop missing returns
    data.dropna(subset = ['RETURN'], axis = 0, how = 'any', inplace = True)

    # Calulate firms now
    data['firms_now'] = data.groupby('DATE')['DW_INSTRUMENT_ID'].transform('count')

    # Calculate coverage
    data['coverage'] = data['firms_now']/data['firms']

    data.drop(['firms_now', 'firms'], axis = 1, inplace = True)
    
    final = pd.DataFrame()  
    final = data.groupby('DATE')['coverage'].mean().reset_index()
    final.rename(columns = {'':'coverage'})
  
    return final


clean = process_data(clean)
all_but_eq = process_data(all_but_eq)
all_but_prc = process_data(all_but_prc)
print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')

unclean = process_data(unclean)
dirty = process_data(dirty)
afp = process_data(afp)


# Save results
today = str(pd.to_datetime("today"))[0:10]

fig, ax = plt.subplots(1, 1)

ax.plot(clean['DATE'], clean['coverage'], label = 'Kruger')
ax.plot(all_but_eq['DATE'], all_but_eq['coverage'], label = 'Kruger Ex Book Equity')
ax.plot(all_but_prc['DATE'], all_but_prc['coverage'], label = 'Kruger Ex Price')
ax.plot(unclean['DATE'], unclean['coverage'], label = 'No Minimum Group Number')
ax.plot(dirty['DATE'], dirty['coverage'], label = 'Full')
ax.plot(afp['DATE'], afp['coverage'], label = 'AFP')

ax.set_xlabel('DATE')
ax.set_ylabel('Coverage Fraction') 
ax.legend(loc = 'best')
ax.set_title('Coverage')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)

fig.savefig(path + '//Data//coverage//coverage_' + str(delay) + '-' + str(signal) + '_cut_' + str(cut) + '_' + today + '.png')

plt.show()
    
print(0.5 * int((time.time() - start_time)/30), ' minutes total')
