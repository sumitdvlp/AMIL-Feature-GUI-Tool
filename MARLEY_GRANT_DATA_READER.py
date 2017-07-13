import pandas as pd
import sklearn
import numpy as np
import csv
import Go

pdata = pd.read_csv("/media/sf_Share/CEDM_51_Updated_features.csv", header=None)
# print(pdata)
df = pd.DataFrame(pdata)
# print(datafrme)

df_merged = pd.DataFrame()
for name, group in df.groupby(0):
    df_merged = df_merged.append(pd.DataFrame(group.values[:, 1:].reshape(1, -1)))

X = df_merged.as_matrix(columns=df_merged.columns[2:])


pdatay = pd.read_csv("/media/sf_Share/y.csv", header=None)
# print(pdata)
dfy = pd.DataFrame(pdatay)
# print(datafrme)

df_mergedy = pd.DataFrame()
for name, group in dfy.groupby(0):
    df_mergedy = df_mergedy.append(pd.DataFrame(group.values[:, 1:].reshape(1, -1)))

y = df_mergedy.as_matrix(columns=df_mergedy.columns[1:2])
np.savetxt("/media/sf_Share/X.csv", X, delimiter=',')
np.savetxt("/media/sf_Share/yres.csv", y, delimiter=',')