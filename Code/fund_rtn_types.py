import custom_functions as cf
import pandas as pd
import time

# Record start time
start_time = time.time()

# Record path
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# What is the cut?
cut = 0.10

# Consider only after 2009
post_2009 = True

# Value weighted
stocks_month_vw = cf.prep_monthly_rtn(path, cut, path + '\\Data\\fundamental_mods\\monthly_fund_rtn_val_wt_cut_' + str(0.1) + '.csv', post_2009)

# Inverse value weighted
stocks_month_ivw = cf.prep_monthly_rtn(path, cut, path + '\\Data\\fundamental_mods\\monthly_fund_rtn_inv_val_wt_cut_' + str(0.1) + '.csv', post_2009)

# Clusters; transitive component equal weighted
stocks_month_clu = cf.prep_monthly_rtn(path, cut, path + '\\Data\\fundamental_mods\\monthly_fund_rtn_cluster_cut_' + str(0.1) + '.csv', post_2009)
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Get Fama-French data
FF, RF = cf.prep_FF_monthly(path)

# Use functions
get_mom_ind = lambda data, signal, wt: cf.get_mom_ind(data, RF, 0, signal, monthly = True, sig_rtn = False, wt = wt)
get_mom_fund = lambda data, signal: cf.get_mom_fund(data, RF, 0, signal, monthly = True, sig_rtn = False)
get_mom_diff = lambda data, signal, wt: cf.get_mom_diff(data, RF, 0, signal, monthly = True, sig_rtn = False, wt = wt)

# VW
ind_mom_rtn1_vw = get_mom_ind(stocks_month_vw, 1, 'vw')
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

fund_mom_rtn1_vw, diff_mom_rtn1_vw = get_mom_fund(stocks_month_vw, 1), get_mom_diff(stocks_month_vw, 1, 'vw') 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

ind_mom_rtn6_vw, fund_mom_rtn6_vw, diff_mom_rtn6_vw = get_mom_ind(stocks_month_vw, 6, 'vw'), get_mom_fund(stocks_month_vw, 6), get_mom_diff(stocks_month_vw, 6, 'vw')
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# IVW
ind_mom_rtn1_ivw = get_mom_ind(stocks_month_ivw, 1, 'ivw') 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

fund_mom_rtn1_ivw, diff_mom_rtn1_ivw = get_mom_fund(stocks_month_ivw, 1), get_mom_diff(stocks_month_ivw, 1, 'ivw') 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

ind_mom_rtn6_ivw, fund_mom_rtn6_ivw, diff_mom_rtn6_ivw = get_mom_ind(stocks_month_ivw, 6, 'ivw'), get_mom_fund(stocks_month_ivw, 6), get_mom_diff(stocks_month_ivw, 6, 'ivw')
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Clusters
fund_mom_rtn1_clu, diff_mom_rtn1_clu = get_mom_fund(stocks_month_clu, 1), get_mom_diff(stocks_month_clu, 1, 'ew') 
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

fund_mom_rtn6_clu, diff_mom_rtn6_clu = get_mom_fund(stocks_month_clu, 6), get_mom_diff(stocks_month_clu, 6, 'ew')
print(0.25 * int((time.time() - start_time)/15), 'minutes so far...')

# Save results
today = str(pd.to_datetime("today"))[0:10]

if post_2009 == True:
    writer = pd.ExcelWriter(path + '//Data//rtn_fund_types//post_2009_rtn_stats_cut_' + str(cut) + '_' + today + '.xlsx')
else:
    writer = pd.ExcelWriter(path + '//Data//rtn_fund_types//rtn_stats_cut_' + str(cut) + '_' + today + '.xlsx')

quants = ['Q1', 'Q5', 'Q5-Q1']
quants_diff = ['Q1', 'Q5', 'Q1-Q5']

# VW
cf.process_rtn(ind_mom_rtn1_vw, quants).to_excel(writer, sheet_name = 'VW ind 1')
cf.process_rtn(fund_mom_rtn1_vw, quants).to_excel(writer, sheet_name = 'VW fund 1')
cf.process_rtn(diff_mom_rtn1_vw, quants_diff).to_excel(writer, sheet_name = 'VW diff 1')

cf.process_rtn(ind_mom_rtn6_vw, quants).to_excel(writer, sheet_name = 'VW ind 6')
cf.process_rtn(fund_mom_rtn6_vw, quants).to_excel(writer, sheet_name = 'VW fund 6')
cf.process_rtn(diff_mom_rtn6_vw, quants_diff).to_excel(writer, sheet_name = 'VW diff 6')

# IVW
cf.process_rtn(ind_mom_rtn1_ivw, quants).to_excel(writer, sheet_name = 'IVW ind 1')
cf.process_rtn(fund_mom_rtn1_ivw, quants).to_excel(writer, sheet_name = 'IVW fund 1')
cf.process_rtn(diff_mom_rtn1_ivw, quants_diff).to_excel(writer, sheet_name = 'IVW diff 1')

cf.process_rtn(ind_mom_rtn6_ivw, quants).to_excel(writer, sheet_name = 'IVW ind 6')
cf.process_rtn(fund_mom_rtn6_ivw, quants).to_excel(writer, sheet_name = 'IVW fund 6')
cf.process_rtn(diff_mom_rtn6_ivw, quants_diff).to_excel(writer, sheet_name = 'IVW diff 6')

# Clusters
cf.process_rtn(fund_mom_rtn1_clu, quants).to_excel(writer, sheet_name = 'Clusters fund 1')
cf.process_rtn(diff_mom_rtn1_clu, quants_diff).to_excel(writer, sheet_name = 'Clusters diff 1')


cf.process_rtn(fund_mom_rtn6_clu, quants).to_excel(writer, sheet_name = 'Clusters fund 6')
cf.process_rtn(diff_mom_rtn6_clu, quants_diff).to_excel(writer, sheet_name = 'Clusters diff 6')

writer.save()

print(0.5 * int((time.time() - start_time)/30), 'minutes total')