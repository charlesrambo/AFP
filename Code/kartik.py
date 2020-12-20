import custom_functions as cf
import pandas as pd

# How long is the signal formation
signal = 6

# What is the delay?
delay = 0

# What is the cut?
cut = 0.10

# What should be the name of the csv file?
output_file = '\\Data\\signals\\diff_signal_' + str(signal) + '_delay_' + str(delay) + '_cut_' + str(cut) + '.csv'

# Where are the data contained? 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Location of return data
rtn_loc = path + "\\Data\\monthly_ret.csv"

# Location of fundamental returns
fund_loc = path + '\\Data\\monthly_fundamental_returns_fuller_cut_' + str(cut) + '.csv'

# Returns the signal
diff_MOM = cf.kartik(rtn_loc = rtn_loc, fund_loc = fund_loc, signal = signal, delay = delay, group_size = 3, other_subset = True, cut = cut)

# Prints the signal to csv
diff_MOM.to_csv(path + '\\' + output_file , index = False)

