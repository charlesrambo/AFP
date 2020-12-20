import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Where are the fundamental returns?
file = path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = None

# Get the month return data
stocks_month = cf.prep_monthly_rtn(path, cut, file, post_2009)

# Get Fama-French data
FF, RF = cf.prep_FF_monthly(path)

# Get duration
duration = (stocks_month['DATE'].dt.year).max() - (stocks_month['DATE'].dt.year).min() + 1

# Use functions
get_mom_ind = lambda signal: cf.get_mom_ind(stocks_month, RF, 0, signal, monthly = True, sig_rtn = True)
get_mom_fund = lambda signal: cf.get_mom_fund(stocks_month, RF, 0, signal, monthly = True, sig_rtn = True)
get_mom_diff = lambda signal: cf.get_mom_diff(stocks_month, RF, 0, signal, monthly = True, sig_rtn = True)
        
# Get data frame
long_short = pd.DataFrame()

# 0-1
ind_mom_rtn1, ind_MOM1 = get_mom_ind(1)
fund_mom_rtn1, fund_MOM1 = get_mom_fund(1) 
diff_mom_rtn1, diff_MOM1  = get_mom_diff(1) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
long_short[['DATE', 'ind_1']] = ind_mom_rtn1[['DATE', 'Q5-Q1']]
long_short['fund_1'] = fund_mom_rtn1['Q5-Q1']
long_short['diff_1'] = diff_mom_rtn1['Q1-Q5']

# 0-6
ind_mom_rtn6, ind_MOM6 = get_mom_ind(6)
fund_mom_rtn6, fund_MOM6 = get_mom_fund(6) 
diff_mom_rtn6, diff_MOM6 = get_mom_diff(6)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
long_short['ind_6'] = ind_mom_rtn6['Q5-Q1']
long_short['fund_6'] = fund_mom_rtn6['Q5-Q1']
long_short['diff_6'] = diff_mom_rtn6['Q1-Q5']


# List of quintiles
quint = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Define function to get cumulative product
get_cum = lambda data, signal: (1 + data[quint]).prod(axis = 0)**(1/(duration - signal/12)) - 1

cum_ind1, cum_fund1, cum_diff1 = get_cum(ind_mom_rtn1, 1), get_cum(fund_mom_rtn1, 1), get_cum(diff_mom_rtn1, 1)

cum_ind6, cum_fund6, cum_diff6 = get_cum(ind_mom_rtn6, 6), get_cum(fund_mom_rtn6, 6), get_cum(diff_mom_rtn6, 6)


index = [1, 6]

ind = pd.concat([cum_ind1, cum_ind6], axis = 1).T
ind.index = index

fund = pd.concat([cum_fund1, cum_fund6], axis = 1).T
fund.index = index

diff = pd.concat([cum_diff1, cum_diff6], axis = 1).T
diff.index = index

print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\wealth_curve_simple\\'
else:
    location = path + '\\Data\\' + folder + '\\'

bars = np.array([1, 2, 3, 4, 5])
width = 0.7
names = ['Industry', 'Fundamental', 'Industry Minus Fundamental']

# Create plots of cumulative returns
fig, axs = plt.subplots(1, 3)
fig.suptitle('One Month Signal')
axs[0].bar(bars, cum_ind1, width, color = '#1f77b4')
axs[1].bar(bars, cum_fund1, width, color = '#ff7f0e')
axs[2].bar(bars, cum_diff1, width, color = '#2ca02c')

for i in range(3):
    axs[i].set_xticks(bars)
    axs[i].set_xticklabels(quint)
    axs[i].set_ylabel('Annualized Return')
    axs[i].set_xlabel('Quintile')
    axs[i].set_title(names[i])
    
fig.set_size_inches(12, 4)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + '//Charts//post_2009_bar_charts_1_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + '//Charts//bar_charts_1_cut_' + str(cut) + '_' + today + '.png')

plt.show()

fig, axs = plt.subplots(1, 3)
fig.suptitle('Six Month Signal')
axs[0].bar(bars, cum_ind6, width, color = '#1f77b4')
axs[1].bar(bars, cum_fund6, width, color = '#ff7f0e')
axs[2].bar(bars, cum_diff6, width, color = '#2ca02c')

for i in range(3):
    axs[i].set_xticks(bars)
    axs[i].set_xticklabels(quint)
    axs[i].set_ylabel('Annualized Return')
    axs[i].set_xlabel('Quintile')
    axs[i].set_title(names[i])
    
fig.set_size_inches(12, 4)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + '//Charts//post_2009_bar_charts_6_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + '//Charts//bar_charts_6_cut_' + str(cut) + '_' + today + '.png')

plt.show()

# Create plots of cumulative returns
fig, axs = plt.subplots(1, 2)

for Q in quint:
    axs[0].plot(diff_mom_rtn1['DATE'], (1 + diff_mom_rtn1[Q]).cumprod() - 1, label = Q)
    axs[1].plot(diff_mom_rtn6['DATE'], (1 + diff_mom_rtn6[Q]).cumprod() - 1, label = Q)

axs[0].set_ylabel('Cumulative Return')
axs[0].set_xlabel('DATE')
axs[0].set_title('One Month')
axs[0].legend(loc = 'upper left')

axs[1].set_ylabel('Cumulative Return')
axs[1].set_xlabel('DATE')
axs[1].set_title('Six Month')
axs[1].legend(loc = 'upper left')

fig.set_size_inches(10, 5)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + '//Charts//post_2009_quintile_cum_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + '//Charts//quintile_cum_cut_' + str(cut) + '_' + today + '.png')

plt.show()

# Get Fama French
FF, _ = cf.prep_FF_monthly(path)

# Merge long_short and FF
long_short = long_short.merge(FF, on = 'DATE')

# Create plots of cumulative returns
fig, axs = plt.subplots(1, 2)

axs[0].plot(long_short['DATE'], (1 + long_short['ind_1']).cumprod() - 1, label = 'Industry')
axs[0].plot(long_short['DATE'], (1 + long_short['fund_1']).cumprod() - 1, label = 'Fundamental')
axs[0].plot(long_short['DATE'], (1 + long_short['diff_1']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[0].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[0].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[0].set_ylabel('Cumulative Return')
axs[0].set_xlabel('DATE')
axs[0].legend(loc = 'upper left')
axs[0].set_title('One Month')

axs[1].plot(long_short['DATE'], (1 + long_short['ind_6']).cumprod() - 1, label = 'Industry')
axs[1].plot(long_short['DATE'], (1 + long_short['fund_6']).cumprod() - 1, label = 'Fundamental')
axs[1].plot(long_short['DATE'], (1 + long_short['diff_6']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[1].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[1].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[1].set_ylabel('Cumulative Return')
axs[1].set_xlabel('DATE')
axs[1].legend(loc = 'upper left')
axs[1].set_title('Six Month')


fig.set_size_inches(10, 5)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + '//Charts//post_2009_long_short_cum_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + '//Charts//long_short_cum_cut_' + str(cut) + '_' + today + '.png')

plt.show()

if folder is None:
    if post_2009 == True:
        writer = pd.ExcelWriter(location + 'post_2009_wealth_curve_rtns_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        writer = pd.ExcelWriter(location + 'wealth_curve_rtns_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    if post_2009 == True:
        writer = pd.ExcelWriter(location + 'Other\\post_2009_wealth_curve_rtns_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        writer = pd.ExcelWriter(location + 'Other\\wealth_curve_rtns_cut_' + str(cut) + '_' + today + '.xlsx')

# Create data frame to save correlations    
correlation = pd.DataFrame(index = [1, 6])

# Create function which calculates correlation
def quick_corr(MOM):
    temp = stocks_month.merge(MOM, on = ['DATE', 'DW_INSTRUMENT_ID'])
    corr = temp['RETURN'].corr(temp['MOM'], method = 'spearman')
    return corr

MOM1_list = [ind_MOM1, fund_MOM1, diff_MOM1]

MOM6_list = [ind_MOM6, fund_MOM6, diff_MOM6]

names = ['ind_mom', 'fund_mom', 'diff_mom']

for i in range(3):
    correlation.loc[1, names[i]] = quick_corr(MOM1_list[i])
    MOM1_list[i]['MOM'] = MOM1_list[i].groupby('DW_INSTRUMENT_ID')['MOM'].shift(1)
    correlation.loc[1, names[i] + '_lag'] = quick_corr(MOM1_list[i])
    correlation.loc[6, names[i]] = quick_corr(MOM6_list[i])
    MOM6_list[i]['MOM'] = MOM6_list[i].groupby('DW_INSTRUMENT_ID')['MOM'].shift(1)
    correlation.loc[6, names[i] + '_lag'] = quick_corr(MOM6_list[i])

ind_fund_list = ['Q1', 'Q5', 'Q5-Q1']
diff_list = ['Q1', 'Q5', 'Q1-Q5']

cf.process_rtn(ind_mom_rtn1, ind_fund_list).to_excel(writer, sheet_name = '1 ind')
cf.process_rtn(fund_mom_rtn1, ind_fund_list).to_excel(writer, sheet_name = '1 fund')
cf.process_rtn(diff_mom_rtn1, diff_list).to_excel(writer, sheet_name = '1 ind-fund')

cf.process_rtn(ind_mom_rtn6, ind_fund_list).to_excel(writer, sheet_name = '6 ind')
cf.process_rtn(fund_mom_rtn6, ind_fund_list).to_excel(writer, sheet_name = '6 fund')
cf.process_rtn(diff_mom_rtn6, diff_list).to_excel(writer, sheet_name = '6 ind-fund')

ind.to_excel(writer, sheet_name = 'industry cum')
fund.to_excel(writer, sheet_name = 'fundamental cum')
diff.to_excel(writer, sheet_name = 'industry minus fundamental cum')

correlation.to_excel(writer, sheet_name = 'correlations sig')

writer.save()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), ' minutes in total.')
