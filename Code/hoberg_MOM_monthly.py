import pandas as pd
import matplotlib.pyplot as plt
import time
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10
# Consider only after 2009
post_2009 = False

# Where are the fundamental returns
file = path + '\\Data\\fundamental_mods\\monthly_fund_rtn_inv_val_wt_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = 'fund_inv_val_wt'

# Get the month return data
stocks_month = cf.prep_monthly_rtn(path, cut, file, post_2009)

# Get Fama-French data
FF, RF = cf.prep_FF_monthly(path)

# Use functions
get_mom_ind = lambda delay, signal: cf.get_mom_ind(stocks_month, RF, delay, signal, monthly = True)
get_mom_fund = lambda delay, signal: cf.get_mom_fund(stocks_month, RF, delay, signal, monthly = True)
        
# 0-1
ind_mom_rtn0_1, fund_mom_rtn0_1 = get_mom_ind(0, 1), get_mom_fund(0, 1) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-3
ind_mom_rtn0_3, fund_mom_rtn0_3 = get_mom_ind(0, 3), get_mom_fund(0, 3)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 1-3 
ind_mom_rtn1_3, fund_mom_rtn1_3 = get_mom_ind(1, 3), get_mom_fund(1, 3)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-6
ind_mom_rtn0_6, fund_mom_rtn0_6 = get_mom_ind(0, 6), get_mom_fund(0, 6) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 1-6
ind_mom_rtn1_6, fund_mom_rtn1_6 = get_mom_ind(1, 6), get_mom_fund(1, 6)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-9
ind_mom_rtn0_9, fund_mom_rtn0_9 = get_mom_ind(0, 9), get_mom_fund(0, 9)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...') 

# 1-9
ind_mom_rtn1_9, fund_mom_rtn1_9 = get_mom_ind(1, 9), get_mom_fund(1, 9)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-12
ind_mom_rtn0_12, fund_mom_rtn0_12 = get_mom_ind(0, 12), get_mom_fund(0, 12)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 1-12 
ind_mom_rtn1_12, fund_mom_rtn1_12 = get_mom_ind(1, 12), get_mom_fund(1, 12)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')    

# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\hoberg_MOM\\'
else:
    location = path + '\\Data\\' + folder + '\\'

if post_2009 == True:
    writer = pd.ExcelWriter(location + 'post_2009_monthly_MOM_results_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    writer = pd.ExcelWriter(location + 'monthly_MOM_results_cut_' + str(cut) + '_' + today + '.xlsx')

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

OLS_results = lambda data: cf.OLS_results(data, FF, ['Q1', 'Q5', 'Q5-Q1'])

if post_2009 == True:
    writer = pd.ExcelWriter(location + 'post_2009_monthly_MOM_factor_loading_cut_' + str(cut) + '_' + today + '.xlsx')
else:
     writer = pd.ExcelWriter(location + 'monthly_MOM_factor_loading_cut_' + str(cut) + '_' + today + '.xlsx')

# 0-1
temp = OLS_results(ind_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 industry)')

temp = OLS_results(fund_mom_rtn0_1)
temp[0].to_excel(writer, sheet_name = '0-1 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-1 fundamental)')

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
temp[0].to_excel(writer, sheet_name = '0-9 fundemental')
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
temp[0].to_excel(writer, sheet_name = '0-12 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-12 fundamental)')

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

if post_2009 == True:
    fig.savefig(location + 'Charts\\post_2009_monthly_MOM_results(short)_cut_' + str(cut) + '_' + today + '.png')
else: 
    fig.savefig(location + 'Charts\\monthly_MOM_results(short)_cut_' + str(cut) + '_' + today + '.png')

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

if post_2009 == True:
    fig.savefig(location + 'Charts\\post_2009_monthly_MOM_results(long)_cut_' + str(cut) + '_' + today + '.png')
else:
    fig.savefig(location + 'Charts\\monthly_MOM_results(long)_cut_' + str(cut) + '_' + today + '.png')

plt.show()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes in total')
