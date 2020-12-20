# -*- coding: utf-8 -*-
import pandas as pd
from pandas.tseries.offsets import Week

path = r'C:\\Users\\rambocha\\Desktop\\Intern_UCLA_AFP_2020'

# Upload daily momentum
mom_daily = pd.read_csv(path + '\\Data\\F-F_Momentum_Factor_daily.txt', header = None)

# Split column
mom_daily[['Date', 'Mom']] = mom_daily[0].str.split('  ', n = 2, expand = True)

# Drop original column
mom_daily.drop(0, axis = 1, inplace = True)

# Drop Fama French copyright at the end of document
mom_daily.drop(24689, axis = 0, inplace = True)

# Convert to datetime object
mom_daily['Date'] = pd.to_datetime(mom_daily['Date'], format = "%Y%m%d") 

# Calculate year
mom_daily['Year'] = mom_daily['Date'].dt.year

# What week of the year is it
mom_daily['Week'] = mom_daily['Date'].dt.week

# Convert returns to a decimal
mom_daily['Mom'] = 0.01 * mom_daily['Mom'].astype(float)

# Create function calculate aggregate returns
prod = lambda x: (1 + x).prod() - 1

# Calculate weekly returns
mom_weekly = mom_daily.groupby(['Year', 'Week'])['Mom'].apply(prod).reset_index()

# Convert time information into one expression
mom_weekly['Date'] = 1000 * mom_weekly.Year + 7 * mom_weekly.Week

# Convert to date object 
mom_weekly['Date'] = pd.to_datetime(mom_weekly['Date'].astype(str), format = "%Y%j", errors = 'coerce') 

# Reorder and get needed columns
mom_weekly = mom_weekly[['Date', 'Mom']]

# Shift to Friday
mom_weekly['Date'] = mom_weekly['Date'] + Week(weekday = 4)

# Save to dataframe
mom_weekly.to_csv(path + '\\Data\\F-F_Momentum_Factor_weekly.csv', index = False)
