import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd

#1
df=pd.read_csv('usa_power.csv')
df1=df[['longitude','latitude']]
df1['Industry']='p'

#2
df2=pd.read_csv('rstatp_usa.csv')

x=[]
for point in df2['WKT']:
    x.append(point.split()[1][1:])
df2['longitude']=x
x=[]
for point in df2['WKT']:
    x.append(point.split()[2][:-1])
df2['latitude']=x
df3=df2.drop(['WKT','f_code','nam','soc'],axis=1)
df3=df3.astype('float')
df3['Industry']='t'


df4=pd.read_csv('deposit.csv')
df5=df4.drop(['gid','dep_name','country','state','latitude','longitude','commodity','dep_type',
             'type_detai','model_code','model_name','metallic','citation'],axis=1)
df5.columns=['longitude','latitude']
df5['Industry']='m'
    
df_concat=pd.concat([df1,df3,df5],ignore_index=True)

df_concat.to_csv('df_concat.csv', encoding='utf-8', index=False)
    
    
    

