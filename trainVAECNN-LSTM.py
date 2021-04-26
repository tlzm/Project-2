# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 10:12:05 2020

@author: tianc
"""



import tensorflow as tf

import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM,Lambda,Flatten,Bidirectional,Conv1D,MaxPooling1D,BatchNormalization
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import objectives
from keras import regularizers
from keras.models import load_model

input_path ='model/pretrain_model_v0.h5'
output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv("input/test.csv")
n_turb = train_data['id'].unique().max()

# pick a large window size of 30 cycles
sequence_length = 19

# function to reshape features into (samples, time steps, features) 
def reshapeFeatures(id_df, seq_length, seq_cols):
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
#test_gen = (list(reshapeFeatures(test_data[test_data['id']==id], sequence_length, sequence_cols)) for id in range(1, n_turb + 1))

# generate sequences and convert to numpy array
feat_array = np.concatenate(list(feat_gen)).astype(np.float32)
#test_array = np.concatenate(list(test_gen)).astype(np.float32)
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
#tlabel_gen = [reshapeLabel(test_data[test_data['id']==id]) for id in range(1, n_turb + 1)]
label_array = np.concatenate(label_gen).astype(np.float32)
#tlabel_array = np.concatenate(tlabel_gen).astype(np.float32)
print(label_array.shape)


seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id']==id]) >= sequence_length]
seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values
label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)


# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def exps(y_true, y_pred):
        return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)+0.5*K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) #


def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]


encoder = load_model(input_path)

x = keras.Input(shape=(sequence_length, 6))

x01=LSTM(units=256, return_sequences=True)(x)
x02=Dropout(0.2)(x01)
x03=LSTM(units=64, return_sequences=True)(x02)
x04=Dropout(0.2)(x03)
x05=LSTM(units=16, return_sequences=False)(x04)
x06=Dropout(0.2)(x05)
x07=Dense(units=16, activation='swish')(x06)

x11=Conv1D(filters=128, kernel_size=2, activation='swish',dilation_rate=1)(x)
#x12=BatchNormalization(momentum=0.6)(x11)
#x12=MaxPooling1D(pool_size=2)(x11)
x13=Conv1D(filters=64, kernel_size=2, activation='swish',dilation_rate=1)(x11)
#x14=BatchNormalization(momentum=0.6)(x13)
x15=Conv1D(filters=32, kernel_size=2, activation='swish',dilation_rate=1)(x13)
#x14=MaxPooling1D(pool_size=2)(x13)
x16=Flatten()(x15)
x17=Dense(units=16, activation='swish')(x16)


y=keras.layers.Concatenate(axis= 1)([x07, x17])


fix = keras.Model(x, y)

#Bidirectional
model = Sequential()
model.add(encoder)
model.add(fix)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model.add(Dropout(0.2, name="dropout_4"))
model.add(Dense(units=nb_out))
model.add(Activation("swish", name="activation_0"))
model.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
#'binary_crossentropy'
print(model.summary())

epochs = 500
batch_size = 1024

# fit the network
history = model.fit(feat_array,label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path))

# summarize history for MAE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('model MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_mae.png")

# summarize history for RMSE
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model RMSE')
plt.ylabel('RMSE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_rmse.png")

# summarize history for Loss
fig_acc = plt.figure(figsize=(10, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_regression_loss.png")

