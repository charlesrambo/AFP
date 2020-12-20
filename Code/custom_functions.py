# Create modual for functions
import numpy as np
import pandas as pd
import datetime as dt
import copy
from pandas.tseries.offsets import MonthEnd, Week
import statsmodels.api as sm

def kartik(rtn_loc, fund_loc, signal, delay, group_size = 3, wt = 'ew', other_subset = True, cut = 0.10):
    
    # Import stocks data
    data = pd.read_csv(rtn_loc, usecols = ['DATE', 'DW_INSTRUMENT_ID', 'RETURN', 'MAIN_KEY', 'INDUSTRY', 
                                                                            'PRICE_UNADJUSTED', 'TOT_EQUITY', 'TOTAL_SHARES']) 

    # Load hoberg data or alternative
    hoberg = pd.read_csv(fund_loc)
          
    # Convert DATE column to a date object
    data['DATE'] = pd.to_datetime(data['DATE'], format = "%Y-%m-%d")
    hoberg['DATE'] = pd.to_datetime(hoberg['DATE'], format = "%Y-%m-%d")
    
    # Create year and month columns
    data['year'] = data['DATE'].dt.year
    
    # Drop missing values
    data.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)
        
    # Convert type of MAIN_KEY column
    data['MAIN_KEY'] = data['MAIN_KEY'].astype('int64')
     
    # Merge Hoberg and stocks_month data
    data = data.merge(hoberg, on = ['DATE', 'MAIN_KEY', 'DW_INSTRUMENT_ID'], how = 'left')
    
    del hoberg
   
    # Drop MAIN_KEY
    data.drop('MAIN_KEY', axis = 1, inplace = True)
    
    # Convert to decimal
    data['RETURN'] = 0.01 * data['RETURN']
    data['TNIC_RET'] = 0.01 * data['TNIC_RET'] 
    
    # Winsorize columns
    data = winsorize(data, 'RETURN')
    
        # Create industry return
    data['ind'], data['N*'] = calc_weighted_rtn(data, 'INDUSTRY', wt = 'ew', monthly = True)

    # Subset to just important columns
    stocks_sub = data[['DATE', 'RETURN', 'TNIC_RET', 'ind', 'N', 'N*', 'INDUSTRY', 'DW_INSTRUMENT_ID', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]
              
    # Compute the difference of industry and fundamental returns
    stocks_sub['ind-fund'] = stocks_sub['ind'] - stocks_sub['TNIC_RET']
   
    # Sort values
    stocks_sub.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)
        
    # Determine the valid observations     
    stocks_sub['valid'] = stocks_sub['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == stocks_sub['DATE'] + MonthEnd(0)   
 
    # Determine momentum
    stocks_sub['MOM'] = stocks_sub['ind-fund'].shift(1 + delay).rolling(signal).apply(prod)
        
    # Get rid of the the invalid observations
    stocks_sub.loc[stocks_sub['valid'] == False, 'MOM'] = np.nan
        
    # Remove observations with undefined momentum or fewer than 5 firms in industry or fundamental group
    if other_subset == True:
        good_firm = stocks_sub['MOM'].notna() &  (stocks_sub['TOT_EQUITY'] > 0) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & (stocks_sub['N'] > group_size - 1) & (stocks_sub['N*'] > group_size - 1)
    else:
        good_firm = stocks_sub['MOM'].notna() & (stocks_sub['N'] > group_size - 1) & (stocks_sub['N*'] > group_size - 1)
    
    stocks_sub = stocks_sub.loc[good_firm, :]
        
    # Drop variables that have done their jobs
    stocks_sub.drop(['valid', 'N', 'N*', 'TOT_EQUITY', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)
    
    # Create quantiles; add 1 to avoid zero-indexing confusion
    stocks_sub['quintile'] = stocks_sub[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))
  
    diff_MOM = stocks_sub[['DATE', 'DW_INSTRUMENT_ID', 'MOM']]
        
    return diff_MOM
    

def winsorize(data, column):
    
    # Record sd
    sd = data.groupby('DATE')[column].transform('std')
    
    # Record mean
    mean = data.groupby('DATE')[column].transform('mean')
    
    # Record lower an upper limits
    lower_lim = mean - 3 * sd
    upper_lim = mean + 3 * sd
    
    # Winsorize
    data.loc[data[column] < lower_lim, column] = lower_lim
    data.loc[data[column] > upper_lim, column] = upper_lim
    
    return data
       
def prep_monthly_rtn(path, cut, file = None, post_2009 = False, fill_fund = False):
    
    # Load hoberg data or alternative
    if file is None:   
        hoberg = pd.read_csv(path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv')
    else:
        hoberg = pd.read_csv(file)
          
    # Import stocks data
    stocks_month = pd.read_csv(path + "\\Data\\monthly_ret.csv", usecols = ['DATE', 'DW_INSTRUMENT_ID', 'RETURN', 'MAIN_KEY', 'INDUSTRY', 
                                                                            'PRICE_UNADJUSTED', 'TOT_EQUITY', 'TOTAL_SHARES']) 
    # Convert DATE column to a date object
    stocks_month['DATE'] = pd.to_datetime(stocks_month['DATE'], format = "%Y-%m-%d")
    hoberg['DATE'] = pd.to_datetime(hoberg['DATE'], format = "%Y-%m-%d")
    
    # Create year and month columns
    stocks_month['year'] = stocks_month['DATE'].dt.year
    
    # Subset stocks data to right time peroid
    if post_2009 == True:      
        stocks_month = stocks_month.loc[stocks_month['year'] > 2009, :]      
    else:     
        stocks_month = stocks_month.loc[stocks_month['year'] < 2010, :]
    
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
    
    if fill_fund == True:
        # Fill missing TNIC_RET with 0
        stocks_month['TNIC_RET'] = stocks_month['TNIC_RET'].fillna(0)
    
    # Winsorize columns
    stocks_month = winsorize(stocks_month, 'RETURN')
      
    return stocks_month

def prep_weekly_rtn(path, cut, file = None, post_2009 = False, fill_fund = False):
    
    # Load hoberg data or alternative
    if file is None:   
        hoberg = pd.read_csv(path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv')
    else:
        hoberg = pd.read_csv(file)
   
    # Load weekly return data
    stocks_week = pd.read_csv(path + '\\Data\\univIMIUSWeeklyReturns.csv')

    # Load monthly return data
    columns = pd.read_csv(path + '\\Data\\monthly_ret.csv', usecols = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY', 'INDUSTRY', 'PRICE_UNADJUSTED', 'TOTAL_SHARES', 'TOT_EQUITY'])

    # Drop missing values
    columns.dropna(subset = ['DW_INSTRUMENT_ID', 'MAIN_KEY'], axis = 0, how = 'any', inplace = True)
    
    # Convert date column to datetime object
    stocks_week['DATE'] = pd.to_datetime(stocks_week['DATE'].astype(str), format = "%Y-%m-%d") 
    columns['DATE'] = pd.to_datetime(columns['DATE'].astype(str), format = "%Y-%m-%d") 
    hoberg['DATE'] = pd.to_datetime(hoberg['DATE'].astype(str), format = "%Y-%m-%d")
    
    # Create year and month columns
    stocks_week['year'] = stocks_week['DATE'].dt.year
    stocks_week['month'] = stocks_week['DATE'].dt.month
    
    # Subset stocks data to right time peroid
    if post_2009 == True:      
        stocks_week = stocks_week.loc[stocks_week['year'] > 2009, :]      
    else:        
        stocks_week = stocks_week.loc[stocks_week['year'] < 2010, :]
      
    # For the monthly ones we need to shift the dates a month minus one week forward
    faux_date = columns['DATE'] + dt.timedelta(days = 7) + MonthEnd(0) - dt.timedelta(days = 6) + Week(weekday = 4)
    columns['year'] = faux_date.dt.year
    columns['month'] = faux_date.dt.month

    del faux_date

    # Drop date from columns
    columns.drop('DATE', axis = 1, inplace = True)

    # Convert type of MAIN_KEY column
    columns['MAIN_KEY'] = columns['MAIN_KEY'].astype('int64')

    # Merge the columns with stocks_week
    stocks_week = stocks_week.merge(columns, on = ['year', 'month', 'DW_INSTRUMENT_ID'])

    # Merge clusters with stocks_week on year and MAIN_KEY
    stocks_week = stocks_week.merge(hoberg, on = ['DATE', 'DW_INSTRUMENT_ID', 'MAIN_KEY'], how = 'left')
    
    # Drop MAIN_KEY
    stocks_week.drop('MAIN_KEY', axis = 1, inplace = True)

    # Convert returns to decimal
    stocks_week['RETURN'] = 0.01 * stocks_week['RETURN']
    stocks_week['TNIC_RET'] = 0.01 * stocks_week['TNIC_RET']
    
    if fill_fund == True:
        # Fill missing TNIC_RET with 0
        stocks_week['TNIC_RET'] = stocks_week['TNIC_RET'].fillna(0)
    
    # Winsorize columns
    stocks_week = winsorize(stocks_week, 'RETURN')
    
    return stocks_week

def prep_FF_weekly(path):
    
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

    # Record risk free returns
    RF = FF[['DATE', 'RF']]

    # Drop risk free date from Fama French dataframe
    FF.drop('RF', axis = 1, inplace = True)
    
    return FF, RF

def prep_FF_monthly(path):

    # Load Fama French three factors
    FF3 = pd.read_csv(path + '\\Data\\F-F_Research_Data_Factors.csv')

    # Convert date column to date
    FF3['DATE'] = pd.to_datetime(FF3['DATE'].astype(str), format = '%Y%m') 

    # Make sure data is at the end of the month
    FF3['DATE'] = FF3['DATE'] + MonthEnd(0)

    # Divide columns be 100
    FF3[['Mkt-RF', 'SMB', 'HML', 'RF']] = FF3[['Mkt-RF', 'SMB', 'HML', 'RF']].div(100)

    # Load faux Fama French weekly momentum
    FF_mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_monthly.csv')

    # Convert date column to datetime
    FF_mom['DATE'] = pd.to_datetime(FF_mom['DATE'].astype(str), format = '%Y-%m-%d') 

    # Make sure date at end of month (Excel messed it up and put at first)
    FF_mom['DATE'] = FF_mom['DATE'] + MonthEnd(0)

    # Merge Fama French data
    FF = FF3.merge(FF_mom, on = 'DATE')

    # Save risk-free rate
    RF = FF[['DATE', 'RF']]

    # Drop risk free date from Fama French dataframe
    FF.drop('RF', axis = 1, inplace = True)
    
    return FF, RF

# function will be used within momentum calculations
prod = lambda x: (1 + x).prod() - 1

def calc_weighted_rtn(stock, column, wt, monthly = True):
    
    # Create deep copy so don't modify original data frame
    data = copy.deepcopy(stock)
    
    if (wt == 'vw')|(wt == 'ivw'):
        # Create market equity column
        data['ME'] = data['TOTAL_SHARES'] * data['PRICE_UNADJUSTED']
    
        # Sort values
        data.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

        if wt == 'vw':
        # Shift results; some dates may be more than a month previous
            data['wt'] = data.groupby('DW_INSTRUMENT_ID')['ME'].shift(1)
            
        else:
            data['wt'] = 1/data.groupby('DW_INSTRUMENT_ID')['ME'].shift(1)
            
        # Check if valid
        if monthly == True:
            data['valid'] = data['DATE'].shift(1) +  dt.timedelta(days = 7) + MonthEnd(0) == data['DATE'] + MonthEnd(0)
        else:
            data['valid'] = data['DATE'].shift(1) +  dt.timedelta(days = 7) == data['DATE']
                
        data.loc[data['valid'] == False, 'wt'] = np.nan
            
            # Drop ME and valid flag
        data.drop(['ME', 'valid'], axis = 1, inplace = True)
    else:
        # If EW all weights are 1
        data.loc[data['RETURN'].notna(), 'wt'] = 1
     
    # Collect total ME_lag value
    data['sum'] = data.groupby(['DATE', column])['wt'].transform('sum')

    # Divide ME_lag by sum
    data['wt'] = data['wt']/data['sum']
    
    # Weight returns
    data['RETURN'] = data['RETURN'] * data['wt']
    
    # Calculate weighted average
    data['RETURN'] = data.groupby(['DATE', column])['RETURN'].transform('sum')
    
    # Record number
    data['N*'] = data.groupby(['DATE', column])['DW_INSTRUMENT_ID'].transform('count')

    return data['RETURN'], data['N*']
 

def get_mom_ind(data, riskfree, delay, signal, monthly = True, sig_rtn = False, wt = 'ew'):
    
    # Create industry return
    data['RETURN'], data['N*'] = calc_weighted_rtn(data, 'INDUSTRY', wt, monthly)
    
    # Subset to just important columns
    ind = data[['DATE', 'RETURN', 'INDUSTRY', 'DW_INSTRUMENT_ID', 'N*']]
   
    # Sort values
    ind.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)
        
    # Determine the valid observations
    if monthly == True:       
        ind['valid'] = ind['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == ind['DATE'] + MonthEnd(0)   
    else:
        ind['valid'] = ind['DATE'].shift(delay + signal) + dt.timedelta(days = 7 * (delay + signal)) == ind['DATE']
           
    # Determine momentum
    ind['MOM'] = ind.groupby('DW_INSTRUMENT_ID')['RETURN'].shift(1 + delay).rolling(signal).apply(prod)
        
    # Get rid of the the invalid observations
    ind.loc[ind['valid'] == False, 'MOM'] = np.nan
    
    # Subset dataset
    good_firm = ind['MOM'].notna() & (ind['N*'] > 2) & (data['TOT_EQUITY'] > 0) & (data['PRICE_UNADJUSTED'] >= 5)
    ind = ind.loc[good_firm, :]
        
    # Drop variables that have done their jobs
    ind.drop(['valid', 'N*'], axis = 1, inplace = True)
    
    # Create quantiles; add 1 to avoid zero-indexing confusion
    ind['quintile'] = ind[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))
  
    # if sig_rtn == True, save MOM
    if sig_rtn == True:
        ind_MOM = ind[['DATE', 'DW_INSTRUMENT_ID', 'MOM']]
  
    # Drop missing returns
    ind.dropna(subset = ['RETURN'], axis = 0, how = 'any', inplace = True)
      
    # Calculate equal weighted returns within each quintile                                                                                          
    I = ind.groupby(['DATE', 'quintile'])['RETURN'].mean().reset_index()
    
    # delete ind
    del ind
        
    # Make quintiles the columns
    ind_5 = I.pivot(index = 'DATE', columns = 'quintile', values = 'RETURN').reset_index()
    
    # Rename columns
    ind_5.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace= True)
    
    # Shift month to end
    ind_5['DATE'] = ind_5['DATE'] + MonthEnd(0)
    
    # Merge with risk-free
    ind_5 = ind_5.merge(riskfree, on = 'DATE')
    
    # When Q5 values are missing, replace with Q4 value
    ind_5.loc[ind_5['Q5'].isna(), 'Q5'] =  ind_5.loc[ind_5['Q5'].isna(), 'Q4']
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        ind_5[Q + '-RF'] = ind_5[Q] - ind_5['RF']
        
    # Construct winners minus losers          
    ind_5['Q5-Q1'] = ind_5['Q5'] - ind_5['Q1']
    
    if sig_rtn == True:
        return ind_5, ind_MOM
    else:
        return ind_5

def get_mom_fund(data, riskfree, delay, signal, monthly = True, sig_rtn = False):

    # Subset to important columns
    stocks_sub = data[['DATE', 'RETURN', 'TNIC_RET', 'N', 'PRICE_UNADJUSTED', 'DW_INSTRUMENT_ID', 'TOT_EQUITY']]
   
    # Sort values
    stocks_sub.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)
    
    # Determine the valid observations
    if monthly == True:           
        stocks_sub['valid'] = stocks_sub['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == stocks_sub['DATE'] + MonthEnd(0)     
    else:   
        stocks_sub['valid'] = stocks_sub['DATE'].shift(delay + signal) + dt.timedelta(days = 7 * (delay + signal)) == stocks_sub['DATE']  
                      
    # Determine momentum
    stocks_sub['MOM'] = stocks_sub.groupby('DW_INSTRUMENT_ID')['TNIC_RET'].shift(1 + delay).rolling(signal).apply(prod)
        
    # Get rid of the the invalid observations
    stocks_sub.loc[stocks_sub['valid'] == False, 'MOM'] = np.nan
        
    # Remove observations with undefined momentum or fewer than 5 firms in fundamental group
    good_firm = stocks_sub['MOM'].notna() & (stocks_sub['N'] > 2) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & (stocks_sub['TOT_EQUITY'] > 0)
    stocks_sub = stocks_sub.loc[good_firm, :]
        
    # Drop variables that have done their jobs
    stocks_sub.drop(['valid', 'N', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)
   
    # Create quantiles; add 1 to avoid zero-indexing confusion
    stocks_sub['quintile'] = stocks_sub[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))

    # Drop missing returns
    stocks_sub.dropna(subset = ['RETURN'], axis = 0, how = 'any', inplace = True)

    # If sig_rtn == True, save signal returns
    if sig_rtn == True:
        fund_MOM = stocks_sub[['DATE', 'DW_INSTRUMENT_ID', 'MOM']]        
    
    # Calculate equal weighted returns within each quintile                                                                                          
    F = stocks_sub.groupby(['DATE', 'quintile'])['RETURN'].mean().reset_index()
    
    # Delte stocks_sub
    del stocks_sub
        
    # Make quintiles the columns
    fund_5 = F.pivot(index = 'DATE', columns = 'quintile', values = 'RETURN').reset_index()  
        
    # Rename columns
    fund_5.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace= True)
    
    # Shift month to end
    fund_5['DATE'] = fund_5['DATE'] + MonthEnd(0)
    
    # Merge with risk-free
    fund_5 = fund_5.merge(riskfree, on = 'DATE')
    
    # When Q5 values are missing, replace with Q4 value
    fund_5.loc[fund_5['Q5'].isna(), 'Q5'] =  fund_5.loc[fund_5['Q5'].isna(), 'Q4']
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        fund_5[Q + '-RF'] = fund_5[Q] - fund_5['RF']
    
    # Calculate Q5-Q1
    fund_5['Q5-Q1'] = fund_5['Q5'] - fund_5['Q1']
    
    if sig_rtn == True:
        return fund_5, fund_MOM
    else:
        return fund_5

def get_mom_diff(data, riskfree, delay, signal, monthly = True, sig_rtn = False, wt = 'ew'):
    
    # Create industry return
    data['ind'], data['N*'] = calc_weighted_rtn(data, 'INDUSTRY', wt, monthly)

    # Subset to just important columns
    stocks_sub = data[['DATE', 'RETURN', 'TNIC_RET', 'ind', 'N', 'N*', 'INDUSTRY', 'DW_INSTRUMENT_ID', 'PRICE_UNADJUSTED', 'TOT_EQUITY']]
              
    # Compute the difference of industry and fundamental returns
    stocks_sub['ind-fund'] = stocks_sub['ind'] - stocks_sub['TNIC_RET']
   
    # Sort values
    stocks_sub.sort_values(['DW_INSTRUMENT_ID', 'DATE'], inplace = True)
        
    # Determine the valid observations
    if monthly == True:       
        stocks_sub['valid'] = stocks_sub['DATE'].shift(signal + delay) + dt.timedelta(days = 7) + MonthEnd(signal + delay) == stocks_sub['DATE'] + MonthEnd(0)   
    else:
        stocks_sub['valid'] = stocks_sub['DATE'].shift(delay + signal) + dt.timedelta(days = 7 * (delay + signal)) == stocks_sub['DATE']
           
    # Determine momentum
    stocks_sub['MOM'] = stocks_sub['ind-fund'].shift(1 + delay).rolling(signal).apply(prod)
        
    # Get rid of the the invalid observations
    stocks_sub.loc[stocks_sub['valid'] == False, 'MOM'] = np.nan
        
    # Remove observations with undefined momentum or fewer than 5 firms in industry or fundamental group
    good_firm = stocks_sub['MOM'].notna() &  (stocks_sub['TOT_EQUITY'] > 0) & (stocks_sub['PRICE_UNADJUSTED'] >= 5) & (stocks_sub['N'] > 2) & (stocks_sub['N*'] > 2)
    stocks_sub = stocks_sub.loc[good_firm, :]
        
    # Drop variables that have done their jobs
    stocks_sub.drop(['valid', 'N', 'N*', 'TOT_EQUITY', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)
    
    # Create quantiles; add 1 to avoid zero-indexing confusion
    stocks_sub['quintile'] = stocks_sub[['DATE', 'MOM']].groupby('DATE')['MOM'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))
  
    if sig_rtn == True:
        diff_MOM = stocks_sub[['DATE', 'DW_INSTRUMENT_ID', 'MOM']]
        
    # Drop missing returns
    stocks_sub.dropna(subset = ['RETURN'], axis = 0, how = 'any', inplace = True)
      
    # Calculate equal weighted returns within each quintile                                                                                          
    D = stocks_sub.groupby(['DATE', 'quintile'])['RETURN'].mean().reset_index()
    
    # Delete stocks_sub
    del stocks_sub
        
    # Make quintiles the columns
    D_5 = D.pivot(index = 'DATE', columns = 'quintile', values = 'RETURN').reset_index()
    
    # Rename columns
    D_5.rename(columns = {0:'Q1', 1:'Q2', 2:'Q3', 3:'Q4', 4:'Q5'}, inplace= True)
    
    # Shift month to end
    D_5['DATE'] = D_5['DATE'] + MonthEnd(0)
    
    # Merge with risk-free
    D_5 = D_5.merge(riskfree, on = 'DATE')
    
    # When Q5 values are missing, replace with Q4 value
    D_5.loc[D_5['Q5'].isna(), 'Q5'] =  D_5.loc[D_5['Q5'].isna(), 'Q4']
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        D_5[Q + '-RF'] = D_5[Q] - D_5['RF']
        
    # Construct winners minus losers          
    D_5['Q1-Q5'] = D_5['Q1'] - D_5['Q5']
    
    if sig_rtn == True:
        return D_5, diff_MOM
    else:
        return D_5


def OLS_results(data, FF, items):
    
    output0 = pd.DataFrame()
    
    output1 = np.array([])
    
    data = data.merge(FF, on = 'DATE')
    
    X = data[['Mkt-RF', 'SMB', 'HML', 'MOM']].values
    
    X = sm.add_constant(X)
    
    for i in items:
        
        y = data[i]

        res = sm.OLS(y, X).fit()
        
        temp = pd.concat((res.params, res.tvalues), axis = 1)
        
        temp.rename(index = {'const':'alpha_{' + str(i) + '}', 'x1':'Mkt-RF', 'x2':'SMB', 'x3':'HML', 'x4':'MOM'}, 
                    columns = {0:'beta', 1:'t'}, inplace = True)
                
        output0 = pd.concat([output0, temp])
        
        output1 = np.append(output1, res.rsquared)
    
    output1 = pd.DataFrame(output1, columns = ['R^2'], index = items)
    output1['N'] = len(data)
    
    return output0, output1

# Process returns to save time in Excel
def process_rtn(data, quants):
    
    results = pd.DataFrame(index = quants)
    
    for Q in quants:
        
        results.loc[Q, 'Mean'] = data[Q].mean()
        results.loc[Q, 'SD'] = data[Q].std()
        results.loc[Q, 'P25'] = data[Q].quantile(0.25)
        results.loc[Q, 'Median'] = data[Q].quantile(0.50)
        results.loc[Q, 'P75'] = data[Q].quantile(0.75) 
        results.loc[Q, 'Sharpe'] = (data[Q] - data['RF']).mean()/data[Q].std()
        results.loc[Q, 'N'] = data[Q].count()   
        
    return results.round(3)
