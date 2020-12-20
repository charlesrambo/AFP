import pandas as pd
import datetime as dt
import numpy as np
from pandas.tseries.offsets import MonthEnd, Week
import statsmodels.api as sm
import copy
import time
import matplotlib.pyplot as plt

# Record start time
start_time = time.time()

# What is the cut
cut = 0.15

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load weekly return data
stocks_week = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")

# Load needed columns of monthly return data
columns = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'PRICE_UNADJUSTED', 'TOTAL_SHARES', 'TOT_EQUITY'])

# Convert dates to datetime objects
stocks_week['DATE'] = pd.to_datetime(stocks_week['DATE'], format = "%Y-%m-%d")
columns['DATE'] = pd.to_datetime(columns['DATE'], format = "%Y-%m-%d")

# Only consider observations up to 2009
stocks_week = stocks_week.loc[stocks_week['DATE'].dt.year < 2010, :]

# Create market equity column
columns['ME'] = columns['PRICE_UNADJUSTED'] * columns['TOTAL_SHARES']

# Sort columns
columns.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

# Shift market equity
columns['ME_Lag'] = columns.groupby('DW_INSTRUMENT_ID')['ME'].shift(1)

# Create flag for invalids
columns['valid'] = columns.groupby('DW_INSTRUMENT_ID')['DATE'].shift(1) + dt.timedelta(days = 7) + MonthEnd(0) == columns['DATE'] + MonthEnd(0)

# Remove the invalid observations
columns.loc[columns['valid'] == False, 'ME_Lag'] = np.nan

# Drop columns after they have done their job
columns.drop(['valid', 'ME'], axis = 1, inplace = True)

# Get year and month for weekly data
stocks_week['year'] = stocks_week['DATE'].dt.year
stocks_week['month'] = stocks_week['DATE'].dt.month

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = columns['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
columns['year'] = faux_date.dt.year
columns['month'] = faux_date.dt.month

del faux_date

# Drop date from columns
columns.drop('DATE', axis = 1, inplace = True)

# Merge on year, month, and DW Instrument
stocks_week = stocks_week.merge(columns, on = ['year', 'month', 'DW_INSTRUMENT_ID'])

# Break into terciles by market equity
stocks_week['quant'] = stocks_week.groupby(['DATE'])['ME_Lag'].transform(lambda x: pd.qcut(x, 3, duplicates = 'drop', labels = False))

del columns

# Drop month column since it has already done its job
stocks_week.drop(['month', 'ME_Lag'], axis = 1, inplace = True)

# Drop duplicates
stocks_week.drop_duplicates(inplace = True)

# Drop values missing instrument ID or gvkey
stocks_week.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert returns to a decimal
stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']

# Rename MAIN_KEY to gvkey1 to avoid confusion
stocks_week.rename(columns = {'MAIN_KEY':'gvkey1'}, inplace = True)

# Convert gvkey1 to integer
stocks_week['gvkey1'] = stocks_week['gvkey1'].astype('int64')

# Load Hoberge data
hoberg = pd.read_csv(path + "\\Data\\tnic2_data.txt", sep = '\t')

# Add one to year to map score onto previous and not current year
hoberg['year'] = 1 + hoberg['year']

# Cut Hoberge data by score
hoberg = hoberg.loc[hoberg['score'] > cut, :]

# Drop score column
hoberg.drop('score', axis = 1, inplace = True)

# Load Fama French three factors
FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors_weekly.csv')

# Convert date column to date
FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m%d') + Week(weekday = 4)

# Divide columns be 100
FF3[['Mkt-RF', 'SMB', 'HML']] = FF3[['Mkt-RF', 'SMB', 'HML']].div(100)

# Load faux Fama French weekly momentum
FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_weekly.csv')

# Convert date column to datetime
FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%m/%d/%Y') 

# Merge Fama French data
FF = FF3.merge(FF_mom, on = 'DATE')

# Drop risk free date from Fama French dataframe
FF.drop('RF', axis = 1, inplace = True)

# Delete dataframes from memory
del FF3
del FF_mom

# Get columns for right side of merge
large = stocks_week.loc[stocks_week['quant'] == 2, :]

# Only consider the smallest stocks measured by ME
small = stocks_week.loc[stocks_week['quant'] == 0, :]

# Construct function to do Hoberg calculation
def calculate_stuff(df1, df2):
    
    # Create deep copies
    left = copy.deepcopy(df1)
    right = copy.deepcopy(df2)

    # Drop the quantile column
    left.drop('quant', axis = 1, inplace = True)
    
    # Only use particular columns on right
    right = right[['DATE', 'gvkey1', 'DW_INSTRUMENT_ID', 'RETURN']]

    # Rename columns
    right.rename(columns = {'gvkey1':'gvkey2', 'DW_INSTRUMENT_ID':'DW_INSTRUMENT_ID2', 'RETURN':'RETURN2'}, inplace = True)

    # Merge fundamentals and Hoberg
    result = left.merge(hoberg, on = ['gvkey1', 'year'])

    # Drop year column
    left.drop('year', axis = 1, inplace = True)

    # Merge fundementals and return data
    result = result.merge(right, on = ['DATE','gvkey2'])

    # Compute mequal weighted returns of large caps
    result['fund'] = result.groupby(['DATE', 'DW_INSTRUMENT_ID'])['RETURN2'].transform('mean')

    # Record number of large cap firms in fundemental group
    result['n'] = result.groupby(['DATE','DW_INSTRUMENT_ID'])['DW_INSTRUMENT_ID2'].transform('count')

    # Sort values
    result.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

    # Shift fund back one week
    result['fund'] = result.groupby('DW_INSTRUMENT_ID')['fund'].shift(1)

    # Record valids
    result['valid'] = result.groupby(['DW_INSTRUMENT_ID'])['DATE'].shift(1) + dt.timedelta(days = 7) == result['DATE']

    # Remove invalids
    result.loc[result['valid'] == False, 'fund'] = np.nan

    # Drop the usual suspects
    result = result.loc[(result['PRICE_UNADJUSTED'] >= 5) & (result['n'] > 4), :]

    # Create quintiles
    result['quant'] = result.groupby('DATE')['fund'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))

    # Take the equally weighted return of each quantile
    result = result.groupby(['DATE', 'quant'])['RETURN'].mean().reset_index()

    # Restructure dataframe
    result = result.pivot(index = 'DATE', columns = 'quant', values = 'RETURN').reset_index()

    # Rename columns
    result.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace = True)

    # Calculate Q5-Q1
    result['Q5-Q1'] = result['Q5'] - result['Q1']
    
    return result

# Use Hoberge returns of just largest to predict smallest
small_by_large = calculate_stuff(small, large)

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Use Hoberge returns of just smallest to predict largest
large_by_small = calculate_stuff(large, small)

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Use Hoberge returns of all stocks to predict smallest
small_by_all = calculate_stuff(small, stocks_week)

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Use Hoberge returns of all stocks to predict largest
large_by_all = calculate_stuff(large, stocks_week)

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Delete dataframes that have done their jobs
del small
del large
del stocks_week
del hoberg

# Create OLS
def OLS_results(data):
    
    output0 = pd.DataFrame()
    
    output1 = np.array([])
    
    data = data.merge(FF, on = 'DATE')
    
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    X = sm.add_constant(X)
    
    for i in ['Q1', 'Q5', 'Q5-Q1']:
        
        y = data[i]

        res = sm.OLS(y, X).fit()
        
        temp = pd.concat((res.params, res.tvalues), axis = 1)
        
        temp.rename(index = {'const':'alpha_{' + str(i) +'}', 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
                
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = ['Q1', 'Q5', 'Q5-Q1'])
    
    return output0, output1

# What is today's date
today = str(pd.to_datetime("today"))[0:10]

writer = pd.ExcelWriter(path + '\\Data\\Lo_Mackinlay\\factor_loadings_cut_' + str(cut) +  '_' + today + '.xlsx')

temp = OLS_results(small_by_large)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SL')
temp[1].to_excel(writer, sheet_name = 'R^2 SL')
small_by_large.describe().to_excel(writer, sheet_name = 'Return stats SL')

temp = OLS_results(large_by_small)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LS')
temp[1].to_excel(writer, sheet_name = 'R^2 LS')
large_by_small.describe().to_excel(writer, sheet_name = 'Return stats LS')

temp = OLS_results(small_by_all)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SA')
temp[1].to_excel(writer, sheet_name = 'R^2 SA')
small_by_all.describe().to_excel(writer, sheet_name = 'Return stats SA')

temp = OLS_results(large_by_all)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LA')
temp[1].to_excel(writer, sheet_name = 'R^2 LA')
large_by_all.describe().to_excel(writer, sheet_name = 'Return stats LA')

writer.save()
        
# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(4, 1)

axs[0].plot(small_by_large['DATE'], (1 + small_by_large['Q1']).cumprod() - 1, label = 'Q1')
axs[0].plot(small_by_large['DATE'], (1 + small_by_large['Q5']).cumprod() - 1, label = 'Q5')
axs[0].plot(small_by_large['DATE'], (1 + small_by_large['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[0].legend(loc = "upper left")
axs[0].set_title('Predict small by large')

axs[1].plot(large_by_small['DATE'], (1 + large_by_small['Q1']).cumprod() - 1, label = 'Q1')
axs[1].plot(large_by_small['DATE'], (1 + large_by_small['Q5']).cumprod() - 1, label = 'Q5')
axs[1].plot(large_by_small['DATE'], (1 + large_by_small['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[1].legend(loc = "upper left")
axs[1].set_title('Predict large by small')

axs[2].plot(small_by_all['DATE'], (1 + small_by_all['Q1']).cumprod() - 1, label = 'Q1')
axs[2].plot(small_by_all['DATE'], (1 + small_by_all['Q5']).cumprod() - 1, label = 'Q5')
axs[2].plot(small_by_all['DATE'], (1 + small_by_all['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[2].legend(loc = "upper left")
axs[2].set_title('Predict small by all')

axs[3].plot(large_by_all['DATE'], (1 + large_by_all['Q1']).cumprod() - 1, label = 'Q1')
axs[3].plot(large_by_all['DATE'], (1 + large_by_all['Q5']).cumprod() - 1, label = 'Q5')
axs[3].plot(large_by_all['DATE'], (1 + large_by_all['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[3].legend(loc = "upper left")
axs[3].set_title('Predict large by all')

fig.set_size_inches(8, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\Lo_Mackinlay\\Charts\\returns_' + str(cut) +  '_' + today + '.png')
plt.show()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes total')