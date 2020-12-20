import pandas as pd
import time
import custom_functions as cf

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009?
post_2009 = False

# Is this monthly?
monthly = True

# If file isn't none, in which folder should everything be place?
folder = None

# Where are the fundamental returns?
if monthly == True:
    file = path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'
else:
    file = path + '\\Data\\weekly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'


# Get the month return data and Fama-French data
if monthly == True:
    stocks = cf.prep_monthly_rtn(path, cut, file, post_2009)
    FF, RF = cf.prep_FF_monthly(path)
else:
    stocks = cf.prep_weekly_rtn(path, cut, file, post_2009)    
    FF, RF = cf.prep_FF_weekly(path)


# Use functions
get_mom_ind = lambda signal: cf.get_mom_ind(stocks, RF, 0, signal, monthly = True, sig_rtn = False)
get_mom_fund = lambda signal: cf.get_mom_fund(stocks, RF, 0, signal, monthly = True, sig_rtn = False)
get_mom_diff = lambda signal: cf.get_mom_diff(stocks, RF, 0, signal, monthly = True, sig_rtn = False)
        
# Get momentum returns
ind_mom_rtn1 = get_mom_ind(1)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

fund_mom_rtn1 = get_mom_fund(1) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

diff_mom_rtn1 = get_mom_diff(1) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Get momentum returns
ind_mom_rtn6 = get_mom_ind(6)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

fund_mom_rtn6 = get_mom_fund(6)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')
 
diff_mom_rtn6 = get_mom_diff(6) 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
today = str(pd.to_datetime("today"))[0:10]

if folder is None:
    location = path + '\\Data\\factor_loadings\\'
else:
    location = path + '\\Data\\' + folder + '\\'

ind_fund_list = ['Q1', 'Q5', 'Q5-Q1']     
diff_list = ['Q1', 'Q5', 'Q1-Q5']

if monthly == True:
    if folder is None:
        if post_2009 == True:
            writer = pd.ExcelWriter(location + 'post_2009_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
        else:
            writer = pd.ExcelWriter(location + 'factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        if post_2009 == True:
            writer = pd.ExcelWriter(location + 'Other\\post_2009_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
        else:
            writer = pd.ExcelWriter(location + 'Other\\factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    if folder is None:
        if post_2009 == True:
            writer = pd.ExcelWriter(location + 'post_2009_weekly_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
        else:
            writer = pd.ExcelWriter(location + 'weekly_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
    else:
        if post_2009 == True:
            writer = pd.ExcelWriter(location + 'Other\\post_2009_weekly_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')
        else:
            writer = pd.ExcelWriter(location + 'Other\\weekly_factor_loadings_cut_' + str(cut) + '_' + today + '.xlsx')

# 0-1   
temp = cf.OLS_results(ind_mom_rtn1, FF, ind_fund_list)
temp[0].to_excel(writer, sheet_name = 'Ind (0-1)')
temp[1].to_excel(writer, sheet_name = 'R^2 (ind 0-1)')

temp = cf.OLS_results(fund_mom_rtn1, FF, ind_fund_list)
temp[0].to_excel(writer, sheet_name = 'Fund (0-1)')
temp[1].to_excel(writer, sheet_name = 'R^2 (fund 0-1)')

temp = cf.OLS_results(diff_mom_rtn1, FF, diff_list)
temp[0].to_excel(writer, sheet_name = 'Diff (0-1)')
temp[1].to_excel(writer, sheet_name = 'R^2 (diff 0-1)')

# 0-6
temp = cf.OLS_results(ind_mom_rtn6, FF, ind_fund_list)
temp[0].to_excel(writer, sheet_name = 'Ind (0-6)')
temp[1].to_excel(writer, sheet_name = 'R^2 (ind 0-6)')

temp = cf.OLS_results(fund_mom_rtn6, FF, ind_fund_list)
temp[0].to_excel(writer, sheet_name = 'Fund (0-6)')
temp[1].to_excel(writer, sheet_name = 'R^2 (fund 0-6)')

temp = cf.OLS_results(diff_mom_rtn6, FF, diff_list)
temp[0].to_excel(writer, sheet_name = 'Diff (0-6)')
temp[1].to_excel(writer, sheet_name = 'R^2 (diff 0-6)')

writer.save()