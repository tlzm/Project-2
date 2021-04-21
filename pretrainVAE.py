# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 15:27:04 2020

@author: tianc
"""


import tensorflow as tf

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,Lambda,Flatten,Bidirectional,Conv1D
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import objectives
from Attention import Attention_layer
from keras import regularizers

input_path ='model/pretrain_model_v0.h5'

train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv("input/test.csv")
n_turb = train_data['id'].unique().max()

# pick a large window size of 30 cycles
sequence_length = 31

# function to reshape features into (samples, time steps, features) 
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
    num_elements = data_matrix.shape[0] 
    for start, stop in zip(range(0, num_elements-seq_length+1), range(seq_length, num_elements+1)):
        yield data_matrix[start:stop, :]
        
        
# pick the feature columns 
#sensor_cols = ['s' + str(i) for i in range(1,22)]
#sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
#sequence_cols.extend(sensor_cols) 
sequence_cols = ['s2', 's3','s4', 's7', 's8',
         's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
#2, 3, 4, 7, 8, 9,11, 12, 13, 14, 15, 17, 20 and 21


# generator for the sequences
feat_gen = (list(reshapeFeatures(train_data[train_data['id']==id], sequence_length, sequence_cols)) for id in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)
print("The data set has now shape: {} entries, {} cycles and {} features.".format(feat_array.shape[0],feat_array.shape[1],feat_array.shape[2]))


# function to generate label
def reshapeLabel(id_df, seq_length=sequence_length, label=['RUL']):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length."""
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[seq_length-1: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)

print(label_array.shape)

# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) # 均方根差

def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]




original_dim = 14

latent_dim = 7

intermediate_dim = 11

epsilon_std = 1.0


#my tips:encoding

x = keras.Input(shape=(sequence_length, nb_features))

h = Dense(intermediate_dim, activation='relu')(x)

z_mean = Dense(latent_dim)(h)

z_log_var = Dense(latent_dim)(h)

 

#my tips:Gauss sampling,sample Z

def sampling(args): 

    z_mean, z_log_var = args

    epsilon = K.random_normal(shape=(sequence_length,latent_dim), mean=0.)

    return z_mean + K.exp(z_log_var / 2) * epsilon

 

# note that "output_shape" isn't necessary with the TensorFlow backend

# my tips:get sample z(encoded)

z = Lambda(sampling, output_shape=(sequence_length,latent_dim))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later

decoder_h = Dense(intermediate_dim, activation='relu')

decoder_mean = Dense(original_dim, activation='relu')

h_decoded = decoder_h(z)

x_decoded_mean = decoder_mean(h_decoded)

vae = keras.Model(x, x_decoded_mean)

vae.compile(optimizer=Adam(lr=1e-2), loss='mse')

vae.fit(feat_array, feat_array,epochs=500,batch_size=512,shuffle=True,validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(input_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )
 

# build a model to project inputs on the latent space

encoder = keras.Model(x, z_mean,trainable=False)



encoder.save(input_path)
