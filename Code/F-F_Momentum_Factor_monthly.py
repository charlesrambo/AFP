# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import MonthEnd, Week

path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Upload daily momentum
mom = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor.txt', header = None)

# Split column
mom[['DATE', 'MOM']] = mom[0].str.split('  ', n = 2, expand = True)

# Drop original column
mom.drop(0, axis = 1, inplace = True)

# Convert to datetime object
mom['DATE'] = pd.to_datetime(mom['DATE'], format = "%Y%m") 

# Move date to Friday of the end of the month
mom['DATE'] = mom['DATE'] + MonthEnd(0) - Week(weekday = 4)

# Convert returns to a decimal
mom['MOM'] = 0.01 * mom['MOM'].astype(float)

# Save to dataframe
mom.to_csv(path + '\\Data\\F-F_Momentum_Factor_monthly.csv', index = False)

