# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 10:15:44 2020

@author: tianc
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler

names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3','s4', 's5', 's6', 's7', 's8',
         's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

# read training data
train_data = pd.read_csv('input/TrainSet.txt', sep=" ", header=None)
train_data.drop(train_data.columns[[26, 27]], axis=1, inplace=True) # drop默认删除行，删除列+axis=1
train_data.columns = names

train_data = train_data.sort_values(['id','cycle'])

# read test data
test_data = pd.read_csv('input/TestSet.txt', sep=" ", header=None)
test_data.drop(test_data.columns[[26, 27]], axis=1, inplace=True)
test_data.columns = names

# read ground truth data
truth_df = pd.read_csv('input/TestSet_RUL.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)

print("This is the size of the train dataset: {} entries and {} features".format(train_data.shape[0], train_data.shape[1]))
print("This is the size of the test dataset: {} entries and {} features".format(test_data.shape[0],test_data.shape[1]))
print("This is the size of the truth dataset: {} entries and {} features".format(truth_df.shape[0],truth_df.shape[1]))

n_turb = train_data["id"].unique().max()
n_train, n_features = train_data.shape
print("There is {} turbines in each dataset".format(n_turb))

# Data Labeling - generate column RUL
rul = pd.DataFrame(train_data.groupby('id')['cycle'].max()).reset_index() 
rul.columns = ['id', 'max']
train_data = train_data.merge(rul, on=['id'], how='left')
train_data['RUL'] = train_data['max'] - train_data['cycle']
train_data.drop('max', axis=1, inplace=True)

# generate label columns
w1 = 125 
#w0 = 15
train_data['RUL'] = np.where(train_data['RUL'] >= w1, 125, train_data['RUL'] )
#train_data['label2'] = train_data['label1']
#train_data.loc[train_data['RUL'] <= w0, 'label2'] = 2

# MinMax normalization (from 0 to 1)
train_data['cycle_norm'] = train_data['cycle']
cols_normalize = train_data.columns.difference(['id','cycle','RUL','label1','label2']) # 把除了这些标签的都提取出来
scaler = MinMaxScaler()
#scaler = MaxAbsScaler()
#scaler = StandardScaler()
norm_train_data = pd.DataFrame(scaler.fit_transform(train_data[cols_normalize]),columns=cols_normalize, index=train_data.index) # fit_transform是用来算归一化的
join_data = train_data[train_data.columns.difference(cols_normalize)].join(norm_train_data)
train_data = join_data.reindex(columns = train_data.columns)

print("The size of the train data set is now: {} entries and {} features.".format(train_data.shape[0],train_data.shape[1]))

train_data.to_csv('input/train.csv', encoding='utf-8',index = None) 
print("Train Data saved as input/train.csv")

# MinMax normalization (from 0 to 1)
test_data['cycle_norm'] = test_data['cycle']
norm_test_data = pd.DataFrame(scaler.transform(test_data[cols_normalize]),columns=cols_normalize, index=test_data.index)
test_join_data = test_data[test_data.columns.difference(cols_normalize)].join(norm_test_data)
test_data = test_join_data.reindex(columns = test_data.columns)
test_data = test_data.reset_index(drop=True)

# generate RUL
rul = pd.DataFrame(test_data.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)
test_data = test_data.merge(truth_df, on=['id'], how='left')
test_data['RUL'] = test_data['max'] - test_data['cycle']
test_data.drop('max', axis=1, inplace=True) 

# generate label columns w0 and w1 for test data
test_data['RUL'] = np.where(test_data['RUL'] >= w1, 125, test_data['RUL'] )
#test_data['label2'] = test_data['label1']
#test_data.loc[test_data['RUL'] <= w0, 'label2'] = 2

print("The size of the test data set is now: {} entries and {} features.".format(test_data.shape[0],test_data.shape[1]))

n_turbtest = test_data["id"].unique().max()
print("There is {} turbines in each dataset".format(n_turbtest))

test_data.to_csv('input/test.csv', encoding='utf-8',index = None)
print("Test Data saved as input/test.csv")
