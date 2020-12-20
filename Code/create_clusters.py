import pandas as pd
import numpy as np
import copy
from sklearn.cluster import AgglomerativeClustering

path = r'C:\Users\rambocha\Desktop\Intern_UCLA_AFP_2020'

hoberg = pd.read_csv(path + '\\Data\\tnic2_data.txt', sep = '\t')

clusters = copy.deepcopy(hoberg)
clusters.drop(['gvkey2', 'score'], axis = 1, inplace = True)
clusters.drop_duplicates(inplace= True)
clusters.rename(columns = {'gvkey1':'MAIN_KEY'}, inplace = True)
clusters['cluster'] = np.nan

years = np.unique(hoberg.year)

# Hard cut-off for the minimum hoberg score within same cluster; no less than 0
cut = 0.10

# Pad must be greater than 1
pad = 1

# Linkage type
linkage = 'complete' 


for year in years:
    
    subset = hoberg.loc[hoberg.year == year].drop('year', axis = 1)
    
    dis_mat = subset.pivot(index = 'gvkey1', columns = 'gvkey2', values = 'score')
    dis_mat = pad - dis_mat.fillna(-0.5)
    np.fill_diagonal(dis_mat.values, 0)
    clustering = AgglomerativeClustering(n_clusters = None, affinity = 'precomputed', 
                                           linkage = linkage, distance_threshold = pad - cut)
    results = clustering.fit(dis_mat)
    clusters.loc[clusters.year == year, 'cluster'] = results.labels_
    
# Check; results should be a little smaller than in paper
subset = copy.deepcopy(clusters)
subset['n'] = subset.groupby(['year', 'cluster'])['MAIN_KEY'].transform('count')
subset = subset.loc[subset['n'] > 4, :]
subset['n'].describe()

clusters.to_csv(path + '\Data\Clusters\clusters_cut_' + str(cut) + '_linkage_' + linkage + '.csv', index = False)
