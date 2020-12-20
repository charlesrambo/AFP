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

# 0-3
ind_mom_rtn3, ind_MOM3 = get_mom_ind(3)
fund_mom_rtn3, fund_MOM3 = get_mom_fund(3) 
diff_mom_rtn3, diff_MOM3  = get_mom_diff(3)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
long_short['ind_3'] = ind_mom_rtn3['Q5-Q1']
long_short['fund_3'] = fund_mom_rtn3['Q5-Q1']
long_short['diff_3'] = diff_mom_rtn3['Q1-Q5']

# 0-6
ind_mom_rtn6, ind_MOM6 = get_mom_ind(6)
fund_mom_rtn6, fund_MOM6 = get_mom_fund(6) 
diff_mom_rtn6, diff_MOM6 = get_mom_diff(6)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
long_short['ind_6'] = ind_mom_rtn6['Q5-Q1']
long_short['fund_6'] = fund_mom_rtn6['Q5-Q1']
long_short['diff_6'] = diff_mom_rtn6['Q1-Q5']

# 0-12
ind_mom_rtn12, ind_MOM12 = get_mom_ind(12)
fund_mom_rtn12, fund_MOM12 = get_mom_fund(12) 
diff_mom_rtn12, diff_MOM12 = get_mom_diff(12)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...') 

# Save results
long_short['ind_12'] = ind_mom_rtn12['Q5-Q1']
long_short['fund_12'] = fund_mom_rtn12['Q5-Q1']
long_short['diff_12'] = diff_mom_rtn12['Q1-Q5']

# List of quintiles
quint = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Define function to get cumulative product
get_cum = lambda data, signal: (1 + data[quint]).prod(axis = 0)**(1/(duration - signal/12)) - 1


cum_ind1, cum_fund1, cum_diff1 = get_cum(ind_mom_rtn1, 1), get_cum(fund_mom_rtn1, 1), get_cum(diff_mom_rtn1, 1)

cum_ind3, cum_fund3, cum_diff3 = get_cum(ind_mom_rtn3, 3), get_cum(fund_mom_rtn3, 3), get_cum(diff_mom_rtn3, 3)

cum_ind6, cum_fund6, cum_diff6 = get_cum(ind_mom_rtn6, 6), get_cum(fund_mom_rtn6, 6), get_cum(diff_mom_rtn6, 6)

cum_ind12, cum_fund12, cum_diff12 = get_cum(ind_mom_rtn12, 12), get_cum(fund_mom_rtn12, 12), get_cum(diff_mom_rtn12, 12)

index = [1, 3, 6, 12]

ind = pd.concat([cum_ind1, cum_ind3, cum_ind6, cum_ind12], axis = 1).T
ind.index = index

fund = pd.concat([cum_fund1, cum_fund3, cum_fund6, cum_fund12], axis = 1).T
fund.index = index

diff = pd.concat([cum_diff1, cum_diff3, cum_diff6, cum_diff12], axis = 1).T
diff.index= index


# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\wealth_curve\\'
else:
    location = path + '\\Data\\' + folder + '\\'

bars = np.array([1, 2, 3, 4, 5])
width = 0.25

# Create plots of cumulative returns
fig, axs = plt.subplots(2, 2)

axs[0, 0].bar(bars - width, cum_ind1, width, label = 'Industry')
axs[0, 0].bar(bars, cum_fund1, width, label = 'Fundamental')
axs[0, 0].bar(bars + width, cum_diff1, width, label = 'Industry Minus Fundamental')
axs[0, 0].set_xticks(bars)
axs[0, 0].set_xticklabels(quint)
axs[0, 0].set_ylabel('Annualized Return')
axs[0, 0].set_xlabel('Quintile')
axs[0, 0].set_title('One Month Momentum')
axs[0, 0].legend(loc = 'best')

axs[1, 0].bar(bars - width, cum_ind3, width, label = 'Industry')
axs[1, 0].bar(bars, cum_fund3, width, label = 'Fundamental')
axs[1, 0].bar(bars + width, cum_diff3, width, label = 'Industry Minus Fundamental')
axs[1, 0].set_xticks(bars)
axs[1, 0].set_xticklabels(quint)
axs[1, 0].set_ylabel('Annualized Return')
axs[1, 0].set_xlabel('Quintile')
axs[1, 0].set_title('Three Month Momentum')
axs[1, 0].legend(loc = 'best')

axs[0, 1].bar(bars - width, cum_ind6, width, label = 'Industry')
axs[0, 1].bar(bars, cum_fund6, width, label = 'Fundamental')
axs[0, 1].bar(bars + width, cum_diff6, width, label = 'Industry Minus Fundamental')
axs[0, 1].set_xticks(bars)
axs[0, 1].set_xticklabels(quint)
axs[0, 1].set_ylabel('Annualized Return')
axs[0, 1].set_xlabel('Quintile')
axs[0, 1].set_title('Six Month Momentum')
axs[0, 1].legend(loc = 'best')

axs[1, 1].bar(bars - width, cum_ind12, width, label = 'Industry')
axs[1, 1].bar(bars, cum_fund12, width, label = 'Fundamental')
axs[1, 1].bar(bars + width, cum_diff12, width, label = 'Industry Minus Fundamental')
axs[1, 1].set_xticks(bars)
axs[1, 1].set_xticklabels(quint)
axs[1, 1].set_ylabel('Annualized Return')
axs[1, 1].set_xlabel('Quintile')
axs[1, 1].set_title('Twelve Month Momentum')
axs[1, 1].legend(loc = 'best')

fig.set_size_inches(10, 10)
fig.tight_layout(pad = 3)

if post_2009 == True:
    fig.savefig(location + '//Charts//post_2009_bar_charts_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + '//Charts//bar_charts_cut_' + str(cut) + '_' + today + '.png')

plt.show()

# Create plots of cumulative returns
fig, axs = plt.subplots(2, 2)

for Q in quint:
    axs[0, 0].plot(diff_mom_rtn1['DATE'], (1 + diff_mom_rtn1[Q]).cumprod() - 1, label = Q)
    axs[1, 0].plot(diff_mom_rtn3['DATE'], (1 + diff_mom_rtn3[Q]).cumprod() - 1, label = Q)
    axs[0, 1].plot(diff_mom_rtn6['DATE'], (1 + diff_mom_rtn6[Q]).cumprod() - 1, label = Q)
    axs[1, 1].plot(diff_mom_rtn12['DATE'], (1 + diff_mom_rtn12[Q]).cumprod() - 1, label = Q)

axs[0, 0].set_ylabel('Cumulative Return')
axs[0, 0].set_xlabel('DATE')
axs[0, 0].legend(loc = 'upper left')
axs[0, 0].set_title('One Month')

axs[1, 0].set_ylabel('Cumulative Return')
axs[1, 0].set_xlabel('DATE')
axs[1, 0].legend(loc = 'upper left')
axs[1, 0].set_title('Three Month')

axs[0, 1].set_ylabel('Cumulative Return')
axs[0, 1].set_xlabel('DATE')
axs[0, 1].legend(loc = 'upper left')
axs[0, 1].set_title('Six Month')

axs[1, 1].set_ylabel('Cumulative Return')
axs[1, 1].set_xlabel('DATE')
axs[1, 1].legend(loc = 'upper left')
axs[1, 1].set_title('Twelve Month')

fig.set_size_inches(10, 10)
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
fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(long_short['DATE'], (1 + long_short['ind_1']).cumprod() - 1, label = 'Industry')
axs[0, 0].plot(long_short['DATE'], (1 + long_short['fund_1']).cumprod() - 1, label = 'Fundamental')
axs[0, 0].plot(long_short['DATE'], (1 + long_short['diff_1']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[0, 0].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[0, 0].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[0, 0].set_ylabel('Cumulative Return')
axs[0, 0].set_xlabel('DATE')
axs[0, 0].legend(loc = 'upper left')
axs[0, 0].set_title('One Month')

axs[1, 0].plot(long_short['DATE'], (1 + long_short['ind_3']).cumprod() - 1, label = 'Industry')
axs[1, 0].plot(long_short['DATE'], (1 + long_short['fund_3']).cumprod() - 1, label = 'Fundamental')
axs[1, 0].plot(long_short['DATE'], (1 + long_short['diff_3']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[1, 0].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[1, 0].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[1, 0].set_ylabel('Cumulative Return')
axs[1, 0].set_xlabel('DATE')
axs[1, 0].legend(loc = 'upper left')
axs[1, 0].set_title('Three Month')

axs[0, 1].plot(long_short['DATE'], (1 + long_short['ind_6']).cumprod() - 1, label = 'Industry')
axs[0, 1].plot(long_short['DATE'], (1 + long_short['fund_6']).cumprod() - 1, label = 'Fundamental')
axs[0, 1].plot(long_short['DATE'], (1 + long_short['diff_6']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[0, 1].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[0, 1].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[0, 1].set_ylabel('Cumulative Return')
axs[0, 1].set_xlabel('DATE')
axs[0, 1].legend(loc = 'upper left')
axs[0, 1].set_title('Six Month')

axs[1, 1].plot(long_short['DATE'], (1 + long_short['ind_12']).cumprod() - 1, label = 'Industry')
axs[1, 1].plot(long_short['DATE'], (1 + long_short['fund_12']).cumprod() - 1, label = 'Fundamental')
axs[1, 1].plot(long_short['DATE'], (1 + long_short['diff_12']).cumprod() - 1, label = 'Industry Minus Fundamental')
axs[1, 1].plot(long_short['DATE'], (1 + long_short['Mkt-RF']).cumprod() - 1, label = 'Mkt-RF')
axs[1, 1].plot(long_short['DATE'], (1 + long_short['MOM']).cumprod() - 1, label = 'Momentum')
axs[1, 1].set_ylabel('Cumulative Return')
axs[1, 1].set_xlabel('DATE')
axs[1, 1].legend(loc = 'upper left')
axs[1, 1].set_title('Twelve Month')

fig.set_size_inches(10, 10)
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
correlation = pd.DataFrame(index = [1, 3, 6, 12])

ind_mom_rtn1.describe().to_excel(writer, sheet_name = '1 ind')
fund_mom_rtn1.describe().to_excel(writer, sheet_name = '1 fund')
diff_mom_rtn1.describe().to_excel(writer, sheet_name = '1 ind-fund')

temp = stocks_month.merge(diff_MOM1, on = ['DATE', 'DW_INSTRUMENT_ID'])
correlation.loc[1, 'corr'] = temp['RETURN'].corr(temp['MOM'], method = 'spearman')

ind_mom_rtn3.describe().to_excel(writer, sheet_name = '3 ind')
fund_mom_rtn3.describe().to_excel(writer, sheet_name = '3 fund')
diff_mom_rtn3.describe().to_excel(writer, sheet_name = '3 ind-fund')

temp = stocks_month.merge(diff_MOM3, on = ['DATE', 'DW_INSTRUMENT_ID'])
correlation.loc[3, 'corr'] = temp['RETURN'].corr(temp['MOM'], method = 'spearman')

ind_mom_rtn6.describe().to_excel(writer, sheet_name = '6 ind')
fund_mom_rtn6.describe().to_excel(writer, sheet_name = '6 fund')
diff_mom_rtn6.describe().to_excel(writer, sheet_name = '6 ind-fund')

temp = stocks_month.merge(diff_MOM6, on = ['DATE', 'DW_INSTRUMENT_ID'])
correlation.loc[6, 'corr'] = temp['RETURN'].corr(temp['MOM'], method = 'spearman')

ind_mom_rtn12.describe().to_excel(writer, sheet_name = '12 ind')
fund_mom_rtn12.describe().to_excel(writer, sheet_name = '12 fund')
diff_mom_rtn12.describe().to_excel(writer, sheet_name = '12 ind-fund')

temp = stocks_month.merge(diff_MOM12, on = ['DATE', 'DW_INSTRUMENT_ID'])
correlation.loc[12, 'corr'] = temp['RETURN'].corr(temp['MOM'], method = 'spearman')

ind.to_excel(writer, sheet_name = 'industry cum')
fund.to_excel(writer, sheet_name = 'fundamental cum')
diff.to_excel(writer, sheet_name = 'industry minus fundamental cum')

correlation.to_excel(writer, sheet_name = 'correlations diff sig')

del temp

writer.save()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), ' minutes in total.')