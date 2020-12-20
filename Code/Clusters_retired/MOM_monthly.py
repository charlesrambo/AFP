import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthEnd
import statsmodels.api as sm
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

# Load return data
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Drop unneeded columns
stocks_month.drop(['DW_INDEX_ID','DW_SOURCE_ID.x', 'TOTAL_SHARES', 'IDX_SHARES', 'FF_ADJUST',
                   'TOTAL_MCAP', 'IDX_MCAP', 'IDX_WEIGHT', 'NAME', 'ISOCODE', 'SIG_NAME',
       'DW_SOURCE_ID.y', 'SECTOR', 'INDUSTRY_GROUP', 'SUB_INDUSTRY', 'FWD_RETURN', 
       'ADJUSTMENT', 'PRICE'], axis = 1, inplace = True)

# Convert data into datetime object
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")

# Make sure DATE column is at the end of the month
stocks_month['DATE'] = stocks_month['DATE'] + MonthEnd(0)

# Create year variable
stocks_month['year'] = stocks_month['DATE'].dt.year

# Only consider observations before 2009
stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]

# Create month variable
stocks_month['month'] = stocks_month['DATE'].dt.month

# Convert returns to decimal
stocks_month['RETURN'] = 0.01 * stocks_month['RETURN']

# Load Fama French three factors
FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors.csv')

# Convert date column to date
FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m') + MonthEnd(0) 

# Divide columns be 100
FF3[['Mkt-RF', 'SMB', 'HML']] = FF3[['Mkt-RF', 'SMB', 'HML']].div(100)

# Load Fama French momentum
FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_monthly.csv')

# Convert date column to datetime
FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%Y-%m-%d') + MonthEnd(0)

# Merge Fama French data
FF = FF3.merge(FF_mom, on = 'DATE')

# Drop risk free date from Fama French dataframe
FF.drop('RF', axis = 1, inplace = True)

# Construct function
prod = lambda x: (1 + x).prod() - 1

def get_mom_ind(delay, signal):
    
    # Only select observations with stock price less than 5 dollars
    good_firm = (stocks_month['TOT_EQUITY'] > 0) & (stocks_month['PRICE_UNADJUSTED'] >= 5)
    stocks_sub = stocks_month.loc[good_firm, ['DATE', 'RETURN', 'INDUSTRY']]
        
    # Compute equal weighted returns
    ind = stocks_sub.groupby(['DATE', 'INDUSTRY'])['RETURN'].mean().reset_index()
        
    # Compute number of clusters
    ind['n'] = stocks_sub.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('count')
   
    # Sort values
    ind.sort_values(['INDUSTRY', 'DATE'], inplace = True)
        
    # Determine the valid observations
    ind['valid'] = ind['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == ind['DATE'] + MonthEnd(0) 
           
    # Determine momentum
    ind['MOM'] = ind.groupby('INDUSTRY')['RETURN'].transform(lambda x: x.shift(1 + delay).rolling(signal).apply(prod))
        
    # Get rid of the the invalid observations
    ind.loc[ind['valid'] == False, 'MOM'] = np.nan
        
    # Remove observations with undefined momentum or fewer than 5 firms in cluster
    ind = ind.loc[ind['MOM'].notna() & (ind['n'] > 4), :]
        
    # Drop variables that have done their jobs
    ind.drop(['valid', 'n'], axis = 1, inplace = True)
        
    # Create quantiles; add 1 to avoid zero-indexing confusion
    ind['quintile'] = 1 + ind[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'raise', labels = False))
        
    # Calculate equal weighted returns within each quintile                                                                                          
    I = ind.groupby(['DATE', 'quintile'])['RETURN'].mean().reset_index()
        
    # Make quintiles the columns
    ind_5 = I.pivot(index = 'DATE', columns = 'quintile', values = 'RETURN').reset_index()  
        
    # Construct winners minus losers          
    ind_5['Q5-Q1'] = ind_5[len(ind_5.columns) - 1] - ind_5[1]
    
    return ind_5

def get_mom_fund(delay, signal):
    
    # Create dataframes to save results
    fund_mom_rtn = pd.DataFrame()
    
    for year in np.unique(stocks_month['year']):
        
        # Only select observations between year t and t + 1, inclusive, and exclude firms with stock price less than 5 dollars
        good_firm = (stocks_month['year'] >= year) & (stocks_month['year'] < year + 2) & (stocks_month['PRICE_UNADJUSTED'] >= 5) & (stocks_month['TOT_EQUITY'] > 0)
        stocks_sub = stocks_month.loc[good_firm, ['DATE', 'year', 'RETURN', 'MAIN_KEY']]
        
        # Subset clusters
        cluster_sub = clusters.loc[clusters['year'] == year, ['MAIN_KEY', 'cluster']]
        
        # Merge the two; only on MAIN_KEY since we want cluster to map onto observations in year t + 1
        stocks_sub = stocks_sub.merge(cluster_sub, on = 'MAIN_KEY')
        
        # Compute equal weighted returns
        fund = stocks_sub.groupby(['DATE', 'cluster'])['RETURN'].mean().reset_index()
        
        # Get year column
        fund['year'] = fund['DATE'].dt.year
        
        # Compute number of clusters
        fund['n'] = stocks_sub.groupby(['DATE', 'cluster'])['RETURN'].transform('count')
        
        # Sort values
        fund.sort_values(['cluster', 'DATE'], inplace = True)
        
        # Determine the valid observations
        fund['valid'] = fund['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == fund['DATE'] + MonthEnd(0)
        fund.loc[fund['valid'] == True, 'valid'] = fund['year'].shift(signal + delay) == year
           
        # Determine momentum
        fund['MOM'] = fund.groupby('cluster')['RETURN'].transform(lambda x: x.shift(1 + delay).rolling(signal + delay).apply(prod))
        
        # Get rid of the the invalid observations
        fund.loc[fund['valid'] == False, 'MOM'] = np.nan
        
        # Remove observations with undefined momentum or fewer than 5 firms in cluster
        fund = fund.loc[fund['MOM'].notna() & (fund['n'] > 4), :]
        
        # Drop variables that have done their jobs
        fund.drop(['valid', 'n', 'year'], axis = 1, inplace = True)
        
        # Create quantiles; add 1 to avoid zero-indexing confusion
        fund['quintile'] = 1 + fund[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'raise', labels = False))
        
        # Calculate equal weighted returns within each quintile                                                                                          
        F = fund.groupby(['DATE', 'quintile'])['RETURN'].mean().reset_index()
        
        # Make quintiles the columns
        fund_5 = F.pivot(index = 'DATE', columns = 'quintile', values = 'RETURN').reset_index()  
        
        # Concatinate with previous results
        fund_mom_rtn = pd.concat([fund_mom_rtn, fund_5])
        
    try:        
        fund_mom_rtn['Q5-Q1'] = fund_mom_rtn[len(fund_mom_rtn.columns) - 1] - fund_mom_rtn[1]
    except:
        fund_mom_rtn['Q5-Q1'] = np.nan
           
    # Print time so far
    print(0.5 * int((time.time() - start_time)/30), 'minutes so far...')
    
    return fund_mom_rtn 
        
ind_mom_rtn0_1, fund_mom_rtn0_1 = get_mom_ind(0, 1), get_mom_fund(0, 1) 

ind_mom_rtn0_3, fund_mom_rtn0_3 = get_mom_ind(0, 3), get_mom_fund(0, 3) 
ind_mom_rtn1_3, fund_mom_rtn1_3 = get_mom_ind(1, 3), get_mom_fund(1, 3)

ind_mom_rtn0_6, fund_mom_rtn0_6 = get_mom_ind(0, 6), get_mom_fund(0, 6) 
ind_mom_rtn1_6, fund_mom_rtn1_6 = get_mom_ind(1, 6), get_mom_fund(1, 6)

ind_mom_rtn0_9, fund_mom_rtn0_9 = get_mom_ind(0, 9), get_mom_fund(0, 9) 
ind_mom_rtn1_9, fund_mom_rtn1_9 = get_mom_ind(1, 9), get_mom_fund(1, 9)

ind_mom_rtn0_12, fund_mom_rtn0_12 = get_mom_ind(0, 12), get_mom_fund(0, 12) 
ind_mom_rtn1_12, fund_mom_rtn1_12 = get_mom_ind(1, 12), get_mom_fund(1, 12)    

# Save results
today = str(pd.to_datetime("today"))[0:10]
writer = pd.ExcelWriter(path + '\\Data\\clusters_MOM\\monthly_MOM_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

# 0-1
ind_mom_rtn0_1.describe().to_excel(writer, sheet_name = '0-1 ind MOM rtn')
fund_mom_rtn0_1.describe().to_excel(writer, sheet_name = '0-1 fund MOM rtn')

# 0-3
ind_mom_rtn0_3.describe().to_excel(writer, sheet_name = '0-3 ind MOM rtn')
fund_mom_rtn0_3.describe().to_excel(writer, sheet_name = '0-3 fund MOM rtn')

# 1-3
ind_mom_rtn1_3.describe().to_excel(writer, sheet_name = '1-3 ind MOM rtn')
fund_mom_rtn1_3.describe().to_excel(writer, sheet_name = '1-3 fund MOM rtn')

# 0-6
ind_mom_rtn0_6.describe().to_excel(writer, sheet_name = '0-6 ind MOM rtn')
fund_mom_rtn0_6.describe().to_excel(writer, sheet_name = '0-6 fund MOM rtn')

# 1-6
ind_mom_rtn1_6.describe().to_excel(writer, sheet_name = '1-6 ind MOM rtn')
fund_mom_rtn1_6.describe().to_excel(writer, sheet_name = '1-6 fund MOM rtn')

# 0-9
ind_mom_rtn0_9.describe().to_excel(writer, sheet_name = '0-9 ind MOM rtn')
fund_mom_rtn0_9.describe().to_excel(writer, sheet_name = '0-9 fund MOM rtn')

# 1-9
ind_mom_rtn0_9.describe().to_excel(writer, sheet_name = '1-9 ind MOM rtn')
fund_mom_rtn0_9.describe().to_excel(writer, sheet_name = '1-9 fund MOM rtn')

# 0-12
ind_mom_rtn0_12.describe().to_excel(writer, sheet_name = '0-12 ind MOM rtn')
fund_mom_rtn0_12.describe().to_excel(writer, sheet_name = '0-12 fund MOM rtn')

# 1-12
ind_mom_rtn1_12.describe().to_excel(writer, sheet_name = '1-12 ind MOM rtn')
fund_mom_rtn1_12.describe().to_excel(writer, sheet_name = '1-12 fund MOM rtn')

writer.save()

def OLS_results(quant_rtn):
    
    output0 = pd.DataFrame()
    
    output1 = np.array([])
    
    data = quant_rtn.merge(FF, on = 'DATE')
    
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    X = sm.add_constant(X)
    
    for i in [1, 5, 'Q5-Q1']:
        
        y = data[i]

        res = sm.OLS(y, X).fit()
        
        temp = pd.concat((res.params, res.tvalues), axis = 1)
        
        if i != 'Q5-Q1':
            temp.rename(index = {'const':'alpha_Q' + str(i), 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
        else:
            temp.rename(index = {'const':'alpha_' + str(i), 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
                
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = ['Q1', 'Q5', 'Q5-Q1'])
    
    return output0, output1

writer = pd.ExcelWriter(path + '\\Data\\clusters_MOM\\monthly_MOM_factor_loading_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

# 0-1
temp = OLS_results(ind_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 industry)')

temp = OLS_results(fund_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 fundemental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 fundemental)')

# 0-3
temp = OLS_results(ind_mom_rtn0_3)
temp[0].to_excel(writer, sheet_name = '0-3 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-3 industry)')

temp = OLS_results(fund_mom_rtn0_3)
temp[0].to_excel(writer, sheet_name = '0-3 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-3 fundamental)')

# 1-3
temp = OLS_results(ind_mom_rtn1_3)
temp[0].to_excel(writer, sheet_name = '1-3 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-3 industry)')

temp = OLS_results(fund_mom_rtn1_3)
temp[0].to_excel(writer, sheet_name = '1-3 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-3 fundamental)')

# 0-6
temp = OLS_results(ind_mom_rtn0_6)
temp[0].to_excel(writer, sheet_name = '0-6 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-6 industry)')

temp = OLS_results(fund_mom_rtn0_6)
temp[0].to_excel(writer, sheet_name = '0-6 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-6 fundamental)')

# 1-6
temp = OLS_results(ind_mom_rtn1_6)
temp[0].to_excel(writer, sheet_name = '1-6 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-6 industry)')

temp = OLS_results(fund_mom_rtn1_6)
temp[0].to_excel(writer, sheet_name = '1-6 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-6 fundamental)')

# 0-9
temp = OLS_results(ind_mom_rtn0_9)
temp[0].to_excel(writer, sheet_name = '0-9 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-9 industry)')

temp = OLS_results(fund_mom_rtn0_9)
temp[0].to_excel(writer, sheet_name = '0-9 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-9 fundamental)')

# 1-9
temp = OLS_results(ind_mom_rtn1_9)
temp[0].to_excel(writer, sheet_name = '1-9 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-9 industry)')

temp = OLS_results(fund_mom_rtn1_9)
temp[0].to_excel(writer, sheet_name = '1-9 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-9 fundamental)')

# 0-12
temp = OLS_results(ind_mom_rtn0_12)
temp[0].to_excel(writer, sheet_name = '0-12 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-12 industry)')

temp = OLS_results(fund_mom_rtn0_12)
temp[0].to_excel(writer, sheet_name = '0-12 fundemental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-12 fundemental)')

# 1-12
temp = OLS_results(ind_mom_rtn1_12)
temp[0].to_excel(writer, sheet_name = '1-12 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-12 industry)')

temp = OLS_results(fund_mom_rtn1_12)
temp[0].to_excel(writer, sheet_name = '1-12 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-12 fundamental)')

writer.save()
 

# Create plots of cumulative returns
fig, axs = plt.subplots(3, 1)

axs[0].plot(ind_mom_rtn0_1['DATE'], (1 + ind_mom_rtn0_1['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[0].plot(fund_mom_rtn0_1['DATE'], (1 + fund_mom_rtn0_1['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[0].legend(loc = "upper left")
axs[0].set_title('0-1 Momentum')

axs[1].plot(ind_mom_rtn0_3['DATE'], (1 + ind_mom_rtn0_3['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[1].plot(fund_mom_rtn0_3['DATE'], (1 + fund_mom_rtn0_3['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[1].legend(loc = "upper left")
axs[1].set_title('0-3 Momentum')

axs[2].plot(ind_mom_rtn1_3['DATE'], (1 + ind_mom_rtn1_3['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[2].plot(fund_mom_rtn1_3['DATE'], (1 + fund_mom_rtn1_3['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[2].legend(loc = "upper left")
axs[2].set_title('1-3 Momentum')

fig.set_size_inches(8, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\clusters_MOM\\Charts\\monthly_MOM_results(short)_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.png')
plt.show()

fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(ind_mom_rtn0_6['DATE'], (1 + ind_mom_rtn0_6['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[0, 0].plot(fund_mom_rtn0_6['DATE'], (1 + fund_mom_rtn0_6['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('0-6 Momentum')

axs[0, 1].plot(ind_mom_rtn1_6['DATE'], (1 + ind_mom_rtn1_6['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[0, 1].plot(fund_mom_rtn1_6['DATE'], (1 + fund_mom_rtn1_6['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('1-6 Momentum')

axs[1, 0].plot(ind_mom_rtn0_9['DATE'], (1 + ind_mom_rtn0_9['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[1, 0].plot(fund_mom_rtn0_9['DATE'], (1 + fund_mom_rtn0_9['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('0-9 Momentum')

axs[1, 1].plot(ind_mom_rtn1_9['DATE'], (1 + ind_mom_rtn1_9['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[1, 1].plot(fund_mom_rtn1_9['DATE'], (1 + fund_mom_rtn1_9['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('1-9 Momentum')

axs[2, 0].plot(ind_mom_rtn0_12['DATE'], (1 + ind_mom_rtn0_12['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[2, 0].plot(fund_mom_rtn0_12['DATE'], (1 + fund_mom_rtn0_12['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[2, 0].legend(loc = "upper left")
axs[2, 0].set_title('0-12 Momentum')

axs[2, 1].plot(ind_mom_rtn1_12['DATE'], (1 + ind_mom_rtn1_12['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[2, 1].plot(fund_mom_rtn1_12['DATE'], (1 + fund_mom_rtn1_12['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[2, 1].legend(loc = "upper left")
axs[2, 1].set_title('1-12 Momentum')

fig.set_size_inches(8, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\clusters_MOM\\Charts\\monthly_MOM_results(long)_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.png')
plt.show()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')
