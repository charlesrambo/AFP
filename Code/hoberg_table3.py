import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import time
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009?
post_2009 = True

# Where are the fundamental returns
file = path + '\\Data\\fundamental_mods\\weekly_fund_rtn_inv_val_wt_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = 'fund_inv_val_wt'

# Get the weekly return data
stocks_week = cf.prep_weekly_rtn(path, cut, file, post_2009)

# Load Fama French three factors
FF, RF = cf.prep_FF_weekly(path)

# Compute industry returns
stocks_week['ind'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['RETURN'].transform('mean')

# Winsorize industry returns
stocks_week = cf.winsorize(stocks_week, 'ind')

# Compute number of industries
stocks_week['n_ind'] = stocks_week.groupby(['DATE', 'INDUSTRY'])['DW_INSTRUMENT_ID'].transform('count')

# Construct difference of industry and fundemental returns
stocks_week['ind-fund'] = stocks_week['ind'] - stocks_week['TNIC_RET']


def calc_quant_rtn(n):
    
    stocks_week.sort_values(by = ['DW_INSTRUMENT_ID', 'DATE'], inplace = True)

    # Initialize output
    output = pd.DataFrame()
  
    # Select needed columns
    subset = stocks_week[['DATE', 'DW_INSTRUMENT_ID', 'RETURN', 'ind-fund', 'PRICE_UNADJUSTED', 'N', 'n_ind', 'TOT_EQUITY']]
        
    # Determine which observations are valid
    subset['valid'] = subset['DATE'].shift(n) + dt.timedelta(days = 7 * n) == subset['DATE']
        
    # Group data together and perform shift
    subset['ind-fund'] = subset.groupby('DW_INSTRUMENT_ID')['ind-fund'].shift(n)
        
    # Remove invalid observations
    subset.loc[subset['valid'] == False, 'ind-fund'] = np.nan
        
    # Exclude usual suspects
    good_firm = (subset['N'] > 4) & (subset['n_ind'] > 4) & (subset['PRICE_UNADJUSTED'] >= 5) & (subset['TOT_EQUITY'] > 0)
    subset = subset.loc[good_firm, :]
        
    # Drop columns that has done their jobs
    subset.drop(['valid', 'N', 'n_ind', 'PRICE_UNADJUSTED'], axis = 1, inplace = True)
        
    # Drop na values
    subset.dropna(axis = 0, inplace = True)
   
    # Place observations into quintiles; add 1 to avoid zero-indexing confusion
    subset['quants'] = 1 + subset.groupby('DATE')['ind-fund'].transform(lambda x: pd.qcut(x, 5, duplicates = 'drop', labels = False))
        
    # Compute equal weighted returns for each tercile
    temp = subset.groupby(['DATE', 'quants'])['RETURN'].mean().reset_index()
        
    # Pivot results
    output = temp.pivot(index = 'DATE', columns = 'quants', values = 'RETURN').reset_index()
    
    # Rename columns
    output.rename(columns = {1:'Q1', 2:'Q2', 3:'Q3', 4:'Q4', 5:'Q5'}, inplace= True)
    
    # Merge with risk-free
    output = output.merge(RF, on = 'DATE')
    
    # Calculate excess returns
    for Q in ['Q1', 'Q5']:
        
        output[Q + '-RF'] = output[Q] - output['RF']
        
    # Compute Q1-Q3
    output['Q1-Q5'] = output.loc[:, 'Q1'] - output.loc[:, 'Q5']
    
    return output

# How long has it been so far
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Calculate quantile returns for given lags
quant_rtn_1 = calc_quant_rtn(1)
quant_rtn_2 = calc_quant_rtn(2)
quant_rtn_3 = calc_quant_rtn(3)
quant_rtn_4 = calc_quant_rtn(4)
quant_rtn_5 = calc_quant_rtn(5)
quant_rtn_6 = calc_quant_rtn(6)

OLS_results = lambda data: cf.OLS_results(data, FF, ['Q1', 'Q5', 'Q1-Q5'])

# What is today's date
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\hoberg_table3_results\\'
else:
    location = path + '\\Data\\' + folder + '\\'

# Name file based on if considering post 2009
if post_2009 == True:      
    writer = pd.ExcelWriter(location + 'post_2009_table3_results_cut_' + str(cut) + '_' + today + '.xlsx')   
else:
    writer = pd.ExcelWriter(location + 'table3_results_cut_' + str(cut) + '_' + today + '.xlsx')

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

temp = OLS_results(quant_rtn_5)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 5)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 5)')

temp = OLS_results(quant_rtn_6)
temp[0].to_excel(writer, sheet_name = 'Factor Loadings (n = 6)')
temp[1].to_excel(writer, sheet_name = 'R^2 (n = 6)')

writer.save()
        
# Create plots of cumulative returns; rescale returns to match sampling
fig, axs = plt.subplots(3, 2)

axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 0].plot(quant_rtn_1['DATE'], (1 + quant_rtn_1['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[0, 0].legend(loc = "upper left")
axs[0, 0].set_title('n = 1')

axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[0, 1].plot(quant_rtn_2['DATE'], (1 + quant_rtn_2['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[0, 1].legend(loc = "upper left")
axs[0, 1].set_title('n = 2')

axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 0].plot(quant_rtn_3['DATE'], (1 + quant_rtn_3['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[1, 0].legend(loc = "upper left")
axs[1, 0].set_title('n = 3')

axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[1, 1].plot(quant_rtn_4['DATE'], (1 + quant_rtn_4['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[1, 1].legend(loc = "upper left")
axs[1, 1].set_title('n = 4')

axs[2, 0].plot(quant_rtn_5['DATE'], (1 + quant_rtn_5['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 0].plot(quant_rtn_5['DATE'], (1 + quant_rtn_5['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 0].plot(quant_rtn_5['DATE'], (1 + quant_rtn_5['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 0].legend(loc = "upper left")
axs[2, 0].set_title('n = 5')

axs[2, 1].plot(quant_rtn_6['DATE'], (1 + quant_rtn_6['Q1-RF']).cumprod() - 1, label = 'Excess Q1')
axs[2, 1].plot(quant_rtn_6['DATE'], (1 + quant_rtn_6['Q5-RF']).cumprod() - 1, label = 'Excess Q5')
axs[2, 1].plot(quant_rtn_6['DATE'], (1 + quant_rtn_6['Q1-Q5']).cumprod() - 1, label = 'Q1-Q5')
axs[2, 1].legend(loc = "upper left")
axs[2, 1].set_title('n = 6')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + 'Charts\\post_2009_table3_results_cut_' + str(cut) + '_' + today + '.png')
else:
    fig.savefig(location + 'Charts\\table3_results_cut_' + str(cut) + '_' + today + '.png')

plt.show()

if post_2009 == True:   
    writer = pd.ExcelWriter(location + 'post_2009_returns_cut_' + str(cut) + '_' + today + '.xlsx')    
else:   
    writer = pd.ExcelWriter(location + 'returns_cut_' + str(cut) + '_' + today + '.xlsx')
    

quant_rtn_1.describe().to_excel(writer, sheet_name = 'n = 1')

quant_rtn_2.describe().to_excel(writer, sheet_name = 'n = 2')

quant_rtn_3.describe().to_excel(writer, sheet_name = 'n = 3')

quant_rtn_4.describe().to_excel(writer, sheet_name = 'n = 4')

quant_rtn_5.describe().to_excel(writer, sheet_name = 'n = 5')

quant_rtn_6.describe().to_excel(writer, sheet_name = 'n = 6')

writer.save()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes in total')
