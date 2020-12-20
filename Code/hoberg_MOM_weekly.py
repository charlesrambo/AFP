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
file = path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'

# If file isn't none, in which folder should everything be place?
folder = None

# Get the month return data
stocks_week = cf.prep_weekly_rtn(path, cut, file, post_2009)

# Get Fama-French data
FF, RF = cf.prep_FF_weekly(path)

# Use functions
get_mom_ind = lambda delay, signal: cf.get_mom_ind(stocks_week, RF, delay, signal, monthly = False)
get_mom_fund = lambda delay, signal: cf.get_mom_fund(stocks_week, RF, delay, signal, monthly = False)
       
# 0-1        
ind_mom_rtn0_1, fund_mom_rtn0_1 = get_mom_ind(0, 1), get_mom_fund(0, 1)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 1-1
ind_mom_rtn1_1, fund_mom_rtn1_1 = get_mom_ind(1, 1), get_mom_fund(1, 1)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-2
ind_mom_rtn0_2, fund_mom_rtn0_2 = get_mom_ind(0, 2), get_mom_fund(0, 2)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 1-2
ind_mom_rtn1_2, fund_mom_rtn1_2 = get_mom_ind(1, 2), get_mom_fund(1, 2)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 2-2
ind_mom_rtn2_2, fund_mom_rtn2_2 = get_mom_ind(2, 2), get_mom_fund(2, 2)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# 0-3
ind_mom_rtn0_3, fund_mom_rtn0_3 = get_mom_ind(0, 3), get_mom_fund(0, 3)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\hoberg_MOM\\'
else:
    location = path + '\\Data\\' + folder + '\\'

if post_2009 == True:
    writer = pd.ExcelWriter(location + 'post_2009_weekly_MOM_results_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    writer = pd.ExcelWriter(location + 'weekly_MOM_results_cut_' + str(cut) + '_' + today + '.xlsx')

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

OLS_results = lambda data: cf.OLS_results(data, FF, ['Q1', 'Q5', 'Q5-Q1'])

if post_2009 == True:
    writer = pd.ExcelWriter(location + 'post_2009_weekly_MOM_factor_loading_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    writer = pd.ExcelWriter(location + 'weekly_MOM_factor_loading_cut_' + str(cut) + '_' + today + '.xlsx')

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
temp[1].to_excel(writer, sheet_name = 'R^2 (1-1 industry)')

temp = OLS_results(fund_mom_rtn1_1)
temp[0].to_excel(writer, sheet_name = '1-1 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-1 fundamental)')

# 0-2
temp = OLS_results(ind_mom_rtn0_2)
temp[0].to_excel(writer, sheet_name = '0-2industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-2 industry)')

temp = OLS_results(fund_mom_rtn0_2)
temp[0].to_excel(writer, sheet_name = '0-2 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (0-2 fundamental)')

# 1-2
temp = OLS_results(ind_mom_rtn1_2)
temp[0].to_excel(writer, sheet_name = '1-2 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-2 industry)')

temp = OLS_results(fund_mom_rtn1_2)
temp[0].to_excel(writer, sheet_name = '1-2 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (1-2 fundamental)')

# 2-2
temp = OLS_results(ind_mom_rtn2_2)
temp[0].to_excel(writer, sheet_name = '2-2 industry')
temp[1].to_excel(writer, sheet_name = 'R^2 (2-2 industry)')

temp = OLS_results(fund_mom_rtn2_2)
temp[0].to_excel(writer, sheet_name = '2-2 fundamental')
temp[1].to_excel(writer, sheet_name = 'R^2 (2-2 fundamental)')

# 0-3
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

if post_2009 == True:
    fig.savefig(location + 'Charts\\post_2009_weekly_MOM_results_cut_' + str(cut) + '_' + today + '.png')
else:
    fig.savefig(location + 'Charts\\weekly_MOM_results_cut_' + str(cut) + '_' + today + '.png')

plt.show()

# Print how long it took
print(0.5 * int((time.time() - start_time)/30), 'minutes in total')