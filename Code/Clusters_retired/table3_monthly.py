import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tseries.offsets import MonthEnd
import datetime as dt
import time

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What type of linkage?
linkage ='complete'

# What is the cut?
cut = 0.15

# Load clusters 
clusters = pd.read_csv(path + '\\Data\\Clusters\\clusters_cut_' + str(cut) + '_linkage_' + linkage + '.csv')

# Hoberg score in year Y is matched to stock data in year Y + 1
clusters['year'] = 1 + clusters['year']

# Load monthly return data
stocks_month = pd.read_csv(path + '\\Data\\monthly_ret.csv')

# Convert date column to datetime object
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'].astype(str), format = "%Y-%m-%d") 

# Make sure date is at the end of the month
stocks_month['DATE'] = stocks_month['DATE'] + MonthEnd(0)

# Create year column
stocks_month['year'] = stocks_month['DATE'].dt.year

# Only consider observations before 2010
stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]

# Merge clusters with stocks_week on year and MAIN_KEY
stocks_month = stocks_month.merge(clusters, on = ['year', 'MAIN_KEY'])

# Drop MAIN_KEY and month
stocks_month.drop('MAIN_KEY', axis = 1, inplace = True)

# Convert returns to decimal
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']

# Load Fama French three factors
FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors.csv')

# Convert date column to date
FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m') 

# Make sure data is at the end of the month
FF3['DATE'] = FF3['DATE'] + MonthEnd(0)

# Divide columns be 100
FF3[['Mkt-RF', 'SMB', 'HML']] = FF3[['Mkt-RF', 'SMB', 'HML']].div(100)

# Load faux Fama French weekly momentum
FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_monthly.csv')

# Convert date column to datetime
FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%Y-%m-%d') 

# Make sure date at end of month (Excel messed it up and put at first)
FF_mom['DATE'] = FF_mom['DATE'] + MonthEnd(0)

# Merge Fama French data
FF = FF3.merge(FF_mom, on = 'DATE')

# Drop risk free date from Fama French dataframe
FF.drop('RF', axis = 1, inplace = True)

# Delete dataframes from memory
del FF3
del FF_mom
del clusters

# Compute fundemental returns
stocks_month['fund'] = stocks_month.groupby(['DATE', 'cluster'])['RETURN'].transform('mean')

# Compute number of clusters
stocks_month['n_fund'] = stocks_month.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')

# Compute industry returns
stocks_month['ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Compute number of industries
stocks_month['n_ind'] = stocks_month.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')

# Construct difference of industry and fundemental returns
stocks_month['ind-fund'] = stocks_month['ind'] - stocks_month['fund']


def calc_quant_rtn(n):
    
    stocks_month.sort_values(by = ['year', 'DW_INSTRUMENT_ID', 'DATE'], inplace = True)

    # Initialize output
    output = pd.DataFrame()
  
    # Select needed columns
    subset = stocks_month[['DATE', 'DW_INSTRUMENT_ID', 'RETURN', 'ind-fund', 'PRICE_UNADJUSTED', 'n_fund', 'n_ind', 'TOT_EQUITY']]
        
    # Determine which observations are valid
    subset['valid'] = subset['DATE'].shift(n) + dt.timedelta(days = 7) + MonthEnd(n) == subset['DATE'] + MonthEnd(0)
        
    # Group data together and perform shift
    subset['ind-fund'] = subset.groupby('DW_INSTRUMENT_ID')['ind-fund'].shift(n)
        
    # Remove invalid observations
    subset.loc[subset['valid'] == False, 'ind-fund'] = np.nan
        
    # Exclude usual suspects
    good_firm = (subset['n_fund'] > 4) & (subset['n_ind'] > 4) & (subset['PRICE_UNADJUSTED'] >= 5) & (subset['TOT_EQUITY'] > 0)
    subset = subset.loc[good_firm, :]
        
    # Drop columns that has done their jobs
    subset.drop(['valid', 'n_fund', 'n_ind', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)
        
    # Drop na values
    subset.dropna(axis = 0, inplace = True)
   
    # Place observations into quintiles; add 1 to avoid zero-indexing confusion
    subset['quants'] = 1 + subset.groupby('DATE')['ind-fund'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))
        
    # Compute equal weighted returns for each quantile
    temp = subset.groupby(['DATE', 'quants'])['RETURN'].mean().reset_index()
        
    # Pivot results
    output = temp.pivot(index = 'DATE', columns = 'quants', values = 'RETURN').reset_index()
        
    # Compute Q1-Q3
    output['Q1-Q5'] = output.iloc[:, 1] - output.iloc[:, len(output.columns) - 1]
    
    return output

# Calculate quantile returns for given lags
quant_rtn_1 = calc_quant_rtn(1)
quant_rtn_2 = calc_quant_rtn(2)
quant_rtn_3 = calc_quant_rtn(3)
quant_rtn_4 = calc_quant_rtn(4)


def OLS_results(quant_rtn):
    
    output0 = pd.DataFrame()
    
    output1 = np.array([])
    
    data = quant_rtn.merge(FF, on = 'DATE')
    
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    X = sm.add_constant(X)
    
    for i in [1, 5, 'Q1-Q5']:
        
        y = data[i]

        res = sm.OLS(y, X).fit()
        
        temp = pd.concat((res.params, res.tvalues), axis = 1)
        
        if i != 'Q1-Q5':
            temp.rename(index = {'const':'alpha_Q' + str(i), 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
        else:
            temp.rename(index = {'const':'alpha_' + str(i), 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
                
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = ['Q1', 'Q5', 'Q1-Q5'])
    
    return output0, output1

# What is today's date
today = str(pd.to_datetime("today"))[0:10]

writer = pd.ExcelWriter(path + '\\Data\\clusters_table3_results\\monthly_table3_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

temp = OLS_results(quant_rtn_1)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 1)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 1)')

temp = OLS_results(quant_rtn_2)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 2)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 2)')

temp = OLS_results(quant_rtn_3)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 3)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 3)')

temp = OLS_results(quant_rtn_4)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 4)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 4)')

writer.save()
        
# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1[1]).cumprod() - 1, label = 'Q1')
axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1[5]).cumprod() - 1, label = 'Q5')
axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('n = 1')

axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2[1]).cumprod() - 1, label = 'Q1')
axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2[5]).cumprod() - 1, label = 'Q5')
axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('n = 2')

axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3[1]).cumprod() - 1, label = 'Q1')
axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3[5]).cumprod() - 1, label = 'Q5')
axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('n = 3')

axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4[1]).cumprod() - 1, label = 'Q1')
axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4[5]).cumprod() - 1, label = 'Q5')
axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('n = 4')

fig.set_size_inches(8, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\clusters_table3_results\\Charts\\monthly_table3_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.png')
plt.show()


writer = pd.ExcelWriter(path + '\\Data\\clusters_table3_results\\monthly_returns_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

quant_rtn_1.describe().to_excel(writer, sheet_name = 'n = 1')

quant_rtn_2.describe().to_excel(writer, sheet_name = 'n = 2')

quant_rtn_3.describe().to_excel(writer, sheet_name = 'n = 3')

quant_rtn_4.describe().to_excel(writer, sheet_name = 'n = 4')

writer.save()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')