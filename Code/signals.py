# Create modual for functions
import pandas as pd
import custom_functions as cf 
import time

# Record start time
start_time = time.time()

# Where should the fundamental returns be drawn from?
file = None 

#What should the cut be?
cut = 0.10

# Record path 
path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020' 

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
    
# Winsorize columns
stocks_month = cf.winsorize(stocks_month, 'RETURN')
      
FF, RF = cf.prep_FF_monthly(path)

for signal in [1, 3, 6, 12]:
    _, MOM = cf.get_mom_diff(stocks_month, RF, 0, signal, monthly = True, sig_rtn = True)
    
    MOM.to_csv(path + '\\Data\\signals\\ind_minus_fund_0_' + str(signal) + '_cut_' + str(cut) + '.csv', index = False)

    print(0.25 * int((time.time() - start_time)/15), ' minutes so far...')
    
print(0.5 * int((time.time() - start_time)/30), ' minutes total')
    