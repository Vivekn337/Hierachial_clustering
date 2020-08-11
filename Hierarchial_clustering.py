import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix

df = pd.read_csv('BWGHT.csv')
original = df[['cigtax','faminc','bwght','motheduc']].values
hierar=[0]*original.shape[0]
group = np.array(range(0,original.shape[0]))
fake_index=group
while(len(np.unique(group)) >= 1): #Procedure is repeated until distilled to single group
    zero=[0]*(original.shape[0]-len(group))
    hierar=np.vstack((hierar, np.concatenate((group,np.array(zero))))).astype(int)
    new_data=original[fake_index]
    dist_mat=np.array(distance_matrix(new_data,new_data ))     
    min_value=min([min(element[element !=0]) for element in dist_mat])
    indi = np.where(dist_mat==min_value)
    uni_update = np.unique(np.array(indi))
    fake_index = np.unique(group)
    a=[]
    for i in uni_update:
          a.append(fake_index[i])
    b=min(a)
    for i in uni_update:
      fake_index[i]=b     
    value = new_data[fake_index==b,:].mean(0)
    group=fake_index
    fake_index=np.unique(fake_index)
    original[b]=value

