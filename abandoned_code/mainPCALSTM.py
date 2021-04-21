# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:18:39 2020

@author: 24687
"""


import tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from RBM import RBMhidden,RBMvisible
from keras.models import load_model
import keras.backend as K
from keras.layers import Dense, Dropout, LSTM,BatchNormalization, Activation,Conv1D,multiply,Flatten
from Attention import Attention_layer
from sklearn.decomposition import PCA

output_path = 'model/regression_model_v0.h5'
sequence_length = 30
test_data = pd.read_csv("input/test.csv")

n_turb = test_data['id'].unique().max()

# pick the feature columns 
#sensor_cols = ['s' + str(i) for i in range(1,22)]
#sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
#sequence_cols.extend(sensor_cols)
sequence_cols = ['s2', 's3','s4', 's7', 's8','s9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
#2, 3, 4, 7, 8, 9,11, 12, 13, 14, 15, 17, 20 and 21



def reshapeFeatures(id_df, seq_length, seq_cols):
    """
    Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length.
    An alternative would be to pad sequences so that
    we can use shorter ones.
    
    :param id_df: the data set to modify
    :param seq_length: the length of the window
    :param seq_cols: the columns concerned by the step
    :return: a generator of the sequences
    """
    data_matrix = id_df[seq_cols].values
    pca = PCA(n_components=7)
    principalComponents = pca.fit_transform(data_matrix)
    features = range(pca.n_components_)
    data_matrix= principalComponents    
    num_elements = data_matrix.shape[0] 
    for start, stop in zip(range(num_elements-seq_length, num_elements-seq_length+1), range(num_elements, num_elements+1)):
        yield data_matrix[start:stop, :]
        
 


# generator for the sequences
feat_gen = (list(reshapeFeatures(test_data[test_data['id']==id], sequence_length, sequence_cols)) for id in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)
# generator for the sequences

seq_array_test_last = feat_array


print("This is the shape of the test set: {} turbines, {} cycles and {} features.".format(seq_array_test_last.shape[0], seq_array_test_last.shape[1], seq_array_test_last.shape[2]))

print("There is only {} turbines out of {} as {} turbines didn't have more than {} cycles.".format(seq_array_test_last.shape[0], n_turb, n_turb - seq_array_test_last.shape[0], sequence_length))



def gated_activation(x):

    # Used in PixelCNN and WaveNet

    tanh = Activation('tanh')(x)

    sigmoid = Activation('sigmoid')(x)

    return multiply([tanh, sigmoid])




y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values

label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
        return K.mean(K.exp(K.abs(y_pred - y_true)/10), axis=-1) #

def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)


def RMSE(y_true, y_pred):
    RMSE=0
    c=y_pred-y_true
#    print(c)
    d=c.shape[0]
    for i in range(1,d,1):
            RMSE=RMSE+c[i]*c[i]
    return math.sqrt(RMSE/d)


def scorecalc(y_true, y_pred):
    score=0
    c=y_pred-y_true
    d=c.shape[0]
    for i in range(1,d,1):
        if c[i]<0:
            score=score-1+math.exp(-c[i]/10)
#            print(c[i])
#            print(score)
        else:
            score=score-1+math.exp(c[i]/13)
#            print(c[i])
#            print(score)
    return score

# if best iteration's model was saved then load and use it
if os.path.isfile(output_path):
    estimator = load_model(output_path, custom_objects={'root_mean_squared_error': root_mean_squared_error,
                                                        'score_calc':score_calc,'RMSE':RMSE,'RBMhidden':RBMhidden,
                                                        'RBMvisible':RBMvisible,'gated_activation':gated_activation,
                                                        'Attention_layer':Attention_layer,'exps':exps})

    y_pred_test = estimator.predict(seq_array_test_last)
    
    y_true_test = label_array_test_last
    

    # test metrics
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last,verbose = 2)
    print('\nMSE: {}'.format(scores_test[0]))
    print('\nRMSE: {}'.format(RMSE(y_true_test, y_pred_test)))
    print('\nscore: {}'.format(scorecalc(y_true_test, y_pred_test)))
    
    s1 = ((y_pred_test - y_true_test)**2).sum()
    moy = y_pred_test.mean()
    s2 = ((y_pred_test - moy)**2).sum() 
    s = 1 - s1/s2
    print('\nEfficiency: {}%'.format(s * 100))

    test_set = pd.DataFrame(y_pred_test)
    test_set.to_csv('output/submit_test.csv', index = None)
    
if os.path.isfile(output_path):
    # Plot in blue color the predicted data and in green color the
    # actual data to verify visually the accuracy of the model.
    fig_verify = plt.figure(figsize=(60, 30))
    # plt.plot(y_pred_test, 'ro', color="red", lw=3.0)
    # plt.plot(y_true_test, 'ro', color="blue")
    X = np.arange(1, 101)
    width = 0.35
#    plt.bar(X, np.array(y_pred_test).reshape(100,), width, color='r')
#    plt.bar(X + width, np.array(y_true_test).reshape(100,), width, color='b')
 #   plt.xticks(X)
  #  plt.title('Remaining Useful Life for each turbine')
   # plt.ylabel('RUL')
    #plt.xlabel('Turbine')
    #plt.legend(['predicted', 'actual data'], loc='upper left')
    #plt.show()
    #fig_verify.savefig("output/model_regression_verify.png")
