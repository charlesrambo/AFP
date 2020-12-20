import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.tseries.offsets import Week, MonthEnd
import datetime as dt
import statsmodels.api as sm
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
clusters = pd.read_csv(path + '\\Data\\Clusters\\clusters_cut_' + str(cut) +'_linkage_' + linkage + '.csv')

# Hoberg score in year Y is matched to stock data in year Y + 1
clusters['year'] = 1 + clusters['year']

# Load return data
stocks_week = pd.read_csv(path + "\\Data\\univIMIUSWeeklyReturns.csv")
stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv")

# Convert dates to date time objects
stocks_week['DATE'] = pd.to_datetime(stocks_week['DATE'].astype(str), format = "%Y-%m-%d") 
stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'].astype(str), format = "%Y-%m-%d")

# Introduce year variable
stocks_week['year'] = stocks_week['DATE'].dt.year

# Introduce month variable
stocks_week['month'] = stocks_week['DATE'].dt.month

# For the monthly ones we need to shift the dates a month minus one week forward
faux_date = stocks_month['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
stocks_month['year'] = faux_date.dt.year
stocks_month['year'] = faux_date.dt.year
stocks_month['month'] = faux_date.dt.month

del faux_date

# Only consider observations before 2010
stocks_week = stocks_week.loc[stocks_week['year'] < 2010, :]
stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]

# Select columns for future merge
columns = stocks_month.loc[:, ['year', 'month', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]

# Delete unneeded dataset
del stocks_month 

# Merge with weekly data
stocks_week = stocks_week.merge(columns, on = ['year', 'month', 'DW_INSTRUMENT_ID'])

# Drop the unneeded month column
stocks_week.drop('month', axis = 1, inplace = True)

# Merge on year and MAIN_KEY
stocks_week = stocks_week.merge(clusters, on = ['year', 'MAIN_KEY'])

# Convert returns to decimal
stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']

# Load Fama French three factors
FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors_weekly.csv')

# Convert date column to date
FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m%d') + Week(weekday = 4)

# Divide columns be 100
FF3[['Mkt-RF', 'SMB', 'HML']] = FF3[['Mkt-RF', 'SMB', 'HML']].div(100)

# Load Fama French momentum
FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_weekly.csv')

# Convert date column to datetime
FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%m/%d/%Y') + Week(weekday = 4)

# Merge Fama French data
FF = FF3.merge(FF_mom, on = 'DATE')

# Drop risk free date from Fama French dataframe
FF.drop('RF', axis = 1, inplace = True)

# Construct function
prod = lambda x: (1 + x).prod() - 1

# Create function to calculate returns from industry momentum
def get_mom_ind(delay, signal):
    
    # Only select observations with stock price less than 5 dollars
    good_firm = (stocks_week['TOT_EQUITY'] > 0) & (stocks_week['PRICE_UNADJUSTED'] >= 5)
    stocks_sub = stocks_week.loc[good_firm, ['DATE', 'RETURN', 'DW_INSTRUMENT_ID', 'INDUSTRY']]
        
    # Compute equal weighted returns
    ind = stocks_sub.groupby(['DATE', 'INDUSTRY'])['RETURN'].mean().reset_index()
        
    # Compute number of clusters
    ind['n'] = stocks_sub.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')
   
    # Sort values
    ind.sort_values(['INDUSTRY', 'DATE'], inplace = True)
        
    # Determine the valid observations
    ind['valid'] = ind['DATE'].shift(delay + signal) + dt.timedelta(days = 7 * (delay + signal)) == ind['DATE']  
           
    # Determine momentum
    ind['MOM'] = ind.groupby('INDUSTRY')['RETURN'].transform(lambda x: x.shift(delay + 1).rolling(signal).apply(prod))
        
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

# Create function to calculate returns from fundemental momentum
def get_mom_fund(delay, signal):
    
    # Create dataframes to save results
    fund_mom_rtn = pd.DataFrame()
    
    for year in np.unique(stocks_week['year']):
        
        # Only select observations between year t and t + 1, inclusive, and exclude firms with stock price less than 5 dollars
        good_firm = (stocks_week['year'] >= year) & (stocks_week['year'] < year + 2) & (stocks_week['PRICE_UNADJUSTED'] >= 5) & (stocks_week['TOT_EQUITY'] > 0)
        stocks_sub = stocks_week.loc[good_firm, ['DATE', 'year', 'RETURN', 'MAIN_KEY', 'DW_INSTRUMENT_ID']]
        
        # Subset clusters
        cluster_sub = clusters.loc[clusters['year'] == year, ['MAIN_KEY', 'cluster']]
        
        # Merge the two; only on MAIN_KEY since we want cluster to map onto observations in year t + 1
        stocks_sub = stocks_sub.merge(cluster_sub, on = 'MAIN_KEY')
        
        # Compute equal weighted returns
        fund = stocks_sub.groupby(['DATE', 'cluster'])['RETURN'].mean().reset_index()
        
        # Get year column
        fund['year'] = fund['DATE'].dt.year
        
        # Compute number of clusters
        fund['n'] = stocks_sub.groupby(['DATE', 'cluster'])['DW_INSTRUMENT_ID'].transform('count')
        
        # Sort values
        fund.sort_values(['cluster', 'DATE'], inplace = True)
        
        # Determine the valid observations
        fund['valid'] = fund['DATE'].shift(delay + signal) + dt.timedelta(days = 7 * (delay + signal)) == fund['DATE']  
        fund.loc[fund['valid'] == True, 'valid'] = fund['year'].shift(delay + signal) == year
           
        # Determine momentum
        fund['MOM'] = fund.groupby('cluster')['RETURN'].transform(lambda x: x.shift(1 + delay).rolling(signal).apply(prod))
        
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
ind_mom_rtn1_1, fund_mom_rtn1_1 = get_mom_ind(1, 1), get_mom_fund(1, 1)

ind_mom_rtn0_2, fund_mom_rtn0_2 = get_mom_ind(0, 2), get_mom_fund(0, 2)
ind_mom_rtn1_2, fund_mom_rtn1_2 = get_mom_ind(1, 2), get_mom_fund(1, 2)
ind_mom_rtn2_2, fund_mom_rtn2_2 = get_mom_ind(2, 2), get_mom_fund(2, 2)

ind_mom_rtn0_3, fund_mom_rtn0_3 = get_mom_ind(0, 3), get_mom_fund(0, 3)

# Save results
today = str(pd.to_datetime("today"))[0:10]
writer = pd.ExcelWriter(path + '\\Data\\clusters_MOM\\weekly_MOM_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

# 0-1
ind_mom_rtn0_1.describe().to_excel(writer, sheet_name = '0-1 ind MOM rtn')
fund_mom_rtn0_1.describe().to_excel(writer, sheet_name = '0-1 fund MOM rtn')

# 1-1
ind_mom_rtn1_1.describe().to_excel(writer, sheet_name = '1-1 ind MOM rtn')
fund_mom_rtn1_1.describe().to_excel(writer, sheet_name = '1-1 fund MOM rtn')

# 0-2
ind_mom_rtn0_2.describe().to_excel(writer, sheet_name = '0-2 ind MOM rtn')
fund_mom_rtn0_2.describe().to_excel(writer, sheet_name = '0-2 fund MOM rtn')

# 1-2
ind_mom_rtn1_2.describe().to_excel(writer, sheet_name = '1-2 ind MOM rtn')
fund_mom_rtn1_2.describe().to_excel(writer, sheet_name = '1-2 fund MOM rtn')

# 2-2
ind_mom_rtn2_2.describe().to_excel(writer, sheet_name = '2-2 ind MOM rtn')
fund_mom_rtn2_2.describe().to_excel(writer, sheet_name = '2-2 fund MOM rtn')

# 0-3
ind_mom_rtn0_3.describe().to_excel(writer, sheet_name = '0-3 ind MOM rtn')
fund_mom_rtn0_3.describe().to_excel(writer, sheet_name = '0-3 fund MOM rtn')

writer.save()

def OLS_results(quant_rtn):
    
    # Initialize dataframe to save factor loadings and t-values
    output0 = pd.DataFrame()
    
    # Initialize array to save R^2 values
    output1 = np.array([])
    
    # Merge Fama-French data with momentum returns
    data = quant_rtn.merge(FF, on = 'DATE')
    
    # Select explanitory variables
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    # Add a constant
    X = sm.add_constant(X)
    
    # Loop of quintiles and do OLS each time
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
        
        # Stack results with previous work
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    # Convert array to dataframe
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = ['Q1', 'Q5', 'Q5-Q1'])
    
    return output0, output1

writer = pd.ExcelWriter(path + '\\Data\\clusters_MOM\\weekly_MOM_factor_loading_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.xlsx')

# 0-1
temp = OLS_results(ind_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 industry)')

temp = OLS_results(fund_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 fundamental)')

# 1-1
temp = OLS_results(ind_mom_rtn1_1)
temp[0].to_excel(writer, sheet_name = '1-1 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-1industry)')

temp = OLS_results(fund_mom_rtn1_1)
temp[0].to_excel(writer, sheet_name = '1-1 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-1 fundamental)')

# 2 week n = 0
temp = OLS_results(ind_mom_rtn0_2)
temp[0].to_excel(writer, sheet_name = '0-2 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-2 industry)')

temp = OLS_results(fund_mom_rtn0_2)
temp[0].to_excel(writer, sheet_name = '0-2 fundemental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-2 fundamental)')

# 2 week n = 1
temp = OLS_results(ind_mom_rtn1_2)
temp[0].to_excel(writer, sheet_name = '1-2 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-2 industry)')

temp = OLS_results(fund_mom_rtn1_2)
temp[0].to_excel(writer, sheet_name = '1-2 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-2 fundamental)')

# 2 week n = 2
temp = OLS_results(ind_mom_rtn2_2)
temp[0].to_excel(writer, sheet_name = '2-2 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (2-2 industry)')

temp = OLS_results(fund_mom_rtn2_2)
temp[0].to_excel(writer, sheet_name = '2-2 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (2-2 fundamental)')

# Eleven month
temp = OLS_results(ind_mom_rtn0_3)
temp[0].to_excel(writer, sheet_name = '0-3 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-3 industry)')

temp = OLS_results(fund_mom_rtn0_3)
temp[0].to_excel(writer, sheet_name = '0-3  fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-3 fundamental)')

writer.save()

# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(ind_mom_rtn0_1['DATE'], (1 + ind_mom_rtn0_1['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[0, 0].plot(fund_mom_rtn0_1['DATE'], (1 + fund_mom_rtn0_1['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('0-1 Momentum')

axs[0, 1].plot(ind_mom_rtn1_1['DATE'], (1 + ind_mom_rtn1_1['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[0, 1].plot(fund_mom_rtn1_1['DATE'], (1 + fund_mom_rtn1_1['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('1-1 Momentum')

axs[1, 0].plot(ind_mom_rtn0_2['DATE'], (1 + ind_mom_rtn0_2['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[1, 0].plot(fund_mom_rtn0_2['DATE'], (1 + fund_mom_rtn0_2['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('0-2 Momentum')

axs[1, 1].plot(ind_mom_rtn1_2['DATE'], (1 + ind_mom_rtn1_2['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[1, 1].plot(fund_mom_rtn1_2['DATE'], (1 + fund_mom_rtn1_2['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('1-2 Momentum')

axs[2, 0].plot(ind_mom_rtn2_2['DATE'], (1 + ind_mom_rtn2_2['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[2, 0].plot(fund_mom_rtn2_2['DATE'], (1 + fund_mom_rtn2_2['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[2, 0].legend(loc = "upper left")
axs[2, 0].set_title('2-2 Momentum')

axs[2, 1].plot(ind_mom_rtn0_3['DATE'], (1 + ind_mom_rtn0_3['Q5-Q1']).cumprod() - 1, label = 'Industry')
axs[2, 1].plot(fund_mom_rtn0_3['DATE'], (1 + fund_mom_rtn0_3['Q5-Q1']).cumprod() - 1, label = 'Fundamental')
axs[2, 1].legend(loc = "upper left")
axs[2, 1].set_title('0-3 Momentum')

fig.set_size_inches(8, 10)
fig.tight_layout(pad = 3)
fig.savefig(path + '\\Data\\clusters_MOM\\Charts\\weekly_MOM_results_cut_' + str(cut) + '_linkage_' + linkage + '_' + today + '.png')
plt.show()

# Print how long it took
print(int((time.time() - start_time)/60), 'minutes')