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
cut = 0.10

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Load weekly return data
stocks_week = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")

# Load needed columns of monthly return data
columns = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY', 'PRICE_UNADJUSTED', 'TOTAL_SHARES', 'TOT_EQUITY'])

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
stocks_week.drop(['year', 'month', 'ME_Lag'], axis = 1, inplace = True)

# Drop duplicates
stocks_week.drop_duplicates(inplace = True)

# Drop values missing instrument ID or gvkey
stocks_week.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)

# Convert MAIN_KEY to int64
stocks_week['MAIN_KEY'] = stocks_week['MAIN_KEY'].astype('int64')

# Load fundamental return data
hoberg = pd.read_csv(path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv')

# Convert data column to date obect
hoberg['DATE'] = pd.to_datetime(hoberg['DATE'], format = "%Y-%m-%d")

# Fundamental and return data
stocks_week = stocks_week.merge(hoberg, on = ['DATE', 'MAIN_KEY', 'DW_INSTRUMENT_ID'])

del hoberg

# Convert returns to a decimal
stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']
stocks_week['TNIC_RET'] = 0.01 * stocks_week['TNIC_RET'] 

# Load Fama French three factors
FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors_weekly.csv')

# Convert date column to date
FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m%d') + Week(weekday = 4)

# Divide columns be 100
FF3[['Mkt-RF', 'SMB', 'HML', 'RF']] = FF3[['Mkt-RF', 'SMB', 'HML', 'RF']].div(100)

# Load faux Fama French weekly momentum
FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_weekly.csv')

# Convert date column to datetime
FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%m/%d/%Y') 

# Merge Fama French data
FF = FF3.merge(FF_mom, on = 'DATE')

# Save risk-free return data frame
RF = FF[['DATE', 'RF']]

# Drop risk free date from Fama French dataframe
FF.drop('RF', axis = 1, inplace = True)

# Delete dataframes from memory
del FF3
del FF_mom

# Get columns for right side of merge
large = stocks_week.loc[stocks_week['quant'] == 2, :]

# Drop quant
large.drop('quant', axis = 1, inplace = True)

# Only consider the smallest stocks measured by ME
small = stocks_week.loc[stocks_week['quant'] == 0, :]

# Drop quant
small.drop('quant', axis = 1, inplace = True)

del stocks_week

# Construct function to do Hoberg calculation
def calculate_stuff(data, grouping):
    
    df = copy.deepcopy(data)
    
    
    df = copy.deepcopy(data)
    
    if (grouping == 'INDUSTRY')|(grouping == 'ind-fund'):
        
        df.dropna(subset = ['INDUSTRY'], axis = 0, inplace = True)
        
        df['INDUSTRY'] = df.groupby('INDUSTRY')['RETURN'].transform('mean')
        
        df['N_ind'] = df.groupby('INDUSTRY')['DW_INSTRUMENT_ID'].transform('count')
        
        if grouping == 'ind-fund':   
            df['ind-fund']  =  df['INDUSTRY'] -  df['TNIC_RET']
        else:
            df['N'] = 5   
    else:
        df['N_ind'] = 5


    # Shift fund back one week
    df[grouping] = df.groupby('DW_INSTRUMENT_ID')[grouping].shift(1)

    # Record valids
    df['valid'] = df.groupby(['DW_INSTRUMENT_ID'])['DATE'].shift(1) + dt.timedelta(days = 7) == df['DATE']

    # Remove invalids
    df.loc[df['valid'] == False, grouping] = np.nan

    # Drop the usual suspects
    good_firm = (df['TOT_EQUITY'] > 0) & (df['PRICE_UNADJUSTED'] >= 5) & (df['N'] > 4) & (df['N_ind'] > 4)
    df = df.loc[good_firm, :]
    
    # Drop missing values
    df.dropna(subset = [grouping], axis = 0, inplace = True)

    # Create quintiles
    df['quant'] = df.groupby('DATE')[grouping].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))

    # Take the equally weighted return of each quantile
    df = df.groupby(['DATE', 'quant'])['RETURN'].mean().reset_index()

    # Restructure dataframe
    df = df.pivot(index = 'DATE', columns = 'quant', values = 'RETURN').reset_index()

    # Rename columns
    df.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace = True)
    
    # When Q5 is missing, replace with Q4
    df.loc[df['Q5'].isna(), 'Q5'] = df.loc[df['Q5'].isna(), 'Q4']
    
    # Merge with risk-free
    df = df.merge(RF, on = 'DATE')
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        df[Q + '-RF'] = df[Q] - df['RF']


    # Calculate Q1-Q5
    df['Q1-Q5'] = df['Q1'] - df['Q5']    
        
    # Calculate Q5-Q1
    df['Q5-Q1'] = df['Q5'] - df['Q1']
    
    return df

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Predict the returns of small firms using industry returns
small_by_ind = calculate_stuff(small, 'INDUSTRY')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Predict the returns of small firms using fundamental returns
small_by_fund = calculate_stuff(small, 'TNIC_RET')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Predict the returns of small firms by the difference
small_by_diff = calculate_stuff(small, 'ind-fund')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Predict the returns of large firms using industry returns
large_by_ind = calculate_stuff(large, 'INDUSTRY')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Pridct the returns of large firms using fundamental returns
large_by_fund = calculate_stuff(large, 'TNIC_RET')

print(0.25 * int((time.time() - start_time)/15), 'minutes so far')

# Predict the returns of large firms by the difference
large_by_diff = calculate_stuff(large, 'ind-fund')

# Delete dataframes that have done their jobs
del small
del large

# Create OLS
def OLS_results(data):
    
    output0 = pd.DataFrame()
    
    output1 = np.array([])
    
    data = data.merge(FF, on = 'DATE')
    
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    X = sm.add_constant(X)
    
    for i in ['Q1', 'Q5', 'Q1-Q5', 'Q5-Q1']:
        
        y = data[i]

        res = sm.OLS(y, X).fit()
        
        temp = pd.concat((res.params, res.tvalues), axis = 1)
        
        temp.rename(index = {'const':'alpha_{' + str(i) +'}', 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
                
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = ['Q1', 'Q5', 'Q1-Q5', 'Q5-Q1'])
    output1['N'] = len(data)
    
    return output0, output1

# What is today's date
today = str(pd.to_datetime("today"))[0:10]

writer = pd.ExcelWriter(path + '\\Data\\cap_analysis\\factor_loadings_cut_' + str(cut) +  '_' + today + '.xlsx')

temp = OLS_results(small_by_ind)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SI')
temp[1].to_excel(writer, sheet_name = 'R^2 SI')
small_by_ind.describe().to_excel(writer, sheet_name = 'Return stats SI')

temp = OLS_results(small_by_fund)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SF')
temp[1].to_excel(writer, sheet_name = 'R^2 SF')
small_by_fund.describe().to_excel(writer, sheet_name = 'Return stats SF')

temp = OLS_results(small_by_diff)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings SD')
temp[1].to_excel(writer, sheet_name = 'R^2 SD')
small_by_diff.describe().to_excel(writer, sheet_name = 'Return stats SD')

temp = OLS_results(large_by_ind)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LI')
temp[1].to_excel(writer, sheet_name = 'R^2 LI')
large_by_ind.describe().to_excel(writer, sheet_name = 'Return stats LI')

temp = OLS_results(large_by_fund)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LF')
temp[1].to_excel(writer, sheet_name = 'R^2 LF')
large_by_fund.describe().to_excel(writer, sheet_name = 'Return stats LF')

temp = OLS_results(large_by_diff)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings LD')
temp[1].to_excel(writer, sheet_name = 'R^2 LD')
large_by_diff.describe().to_excel(writer, sheet_name = 'Return stats LD')

writer.save()
        
# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 0].plot(small_by_ind['DATE'], (1 + small_by_ind['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('Predict small firm by industry')

axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 0].plot(small_by_fund['DATE'], (1 + small_by_fund['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('Predict small firm by fundamentals')

axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 0].plot(small_by_diff['DATE'], (1 + small_by_diff['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 0].legend(loc = "upper left")
axs[2, 0].set_title('Predict small firm by industry minus fundamentals')

axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 1].plot(large_by_ind['DATE'], (1 + large_by_ind['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('Predict large firm by industry')

axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 1].plot(large_by_fund['DATE'], (1 + large_by_fund['Q5-Q1']).cumprod() - 1, label = 'Q5-Q1')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('Predict large firm by fundamentals')

axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 1].plot(large_by_diff['DATE'], (1 + large_by_diff['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 1].legend(loc = "upper left")
axs[2, 1].set_title('Predict large firm by industry minus fundamentals')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\cap_analysis\\Charts\\returns_cut_' + str(cut) +  '_' + today + '.png')
plt.show()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes total')
