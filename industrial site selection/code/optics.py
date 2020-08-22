from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import OPTICS

import pandas as pd
import numpy as np

from numpy import sin,cos,arctan2,sqrt,pi # import from numpy
# earth's mean radius = 6,371km
EARTHRADIUS = 6371.0

def getDistanceByHaversine(loc1, loc2):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_denter link description hereecimal,lon_decimal) pairs'''

    lat1 = loc1[1]
    lon1 = loc1[0]
    lat2 = loc2[1]
    lon2 = loc2[0]
    #
    # convert to radians 
    lon1 = lon1 * pi / 180.0
    lon2 = lon2 * pi / 180.0
    lat1 = lat1 * pi / 180.0
    lat2 = lat2 * pi / 180.0
    #
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2.0))**2
    c = 2.0 * arctan2(sqrt(a), sqrt(1.0-a))
    km = EARTHRADIUS * c
    return km

def group_euclid(qw):
    return np.sqrt((qw['latitude']-qw['latitude'].mean())**2+(qw['longitude']-qw['longitude'].mean())**2).max()


df1=pd.read_csv('df_concat.csv')
df2=df1.drop(['Industry'],axis=1)



distance_matrix = squareform(pdist(df2, (lambda u,v: getDistanceByHaversine(u,v))))



db=OPTICS(min_samples=5, metric='precomputed')
y_db = db.fit_predict(distance_matrix)

df1['cluster'] = y_db


uf=df1[['Industry','cluster']].groupby('cluster')

unique_cluster=uf.nunique()

mean_cluster=df1[['longitude','latitude','cluster']].groupby('cluster').mean()

max_distance=df1[['longitude','latitude','cluster']].groupby('cluster').apply(group_euclid)

three_cluster=mean_cluster[unique_cluster['Industry']==3]
print(three_cluster.index)
three_cluster['max_distance']=max_distance[three_cluster.index]

second_cluster=mean_cluster[unique_cluster['Industry']==2]
second_cluster.index
second_cluster['max_distance']=max_distance[second_cluster.index]

first_cluster=mean_cluster[unique_cluster['Industry']==1]
first_cluster.index
first_cluster['max_distance']=max_distance[first_cluster.index]


first_cluster.to_csv('first_cluster_optics.csv', encoding='utf-8', index=False)
second_cluster.to_csv('second_cluster_optics.csv', encoding='utf-8', index=False)
three_cluster.to_csv('three_cluster_optics.csv', encoding='utf-8', index=False)

    





