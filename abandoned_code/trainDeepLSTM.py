# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 19:45:01 2020

@author: 24687
"""

import tensorflow
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/train.csv')
n_turb = train_data['id'].unique().max()


output_path1 = 'model/regression_model_v1.h5'
output_path2 = 'model/regression_model_v2.h5'
output_path3 = 'model/regression_model_v3.h5'
output_path4 = 'model/regression_model_v4.h5'

# pick a large window size of 30 cycles
sequence_length = 19

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
    num_elements = data_matrix.shape[0] # 输出行数
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

def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) # 均方根差

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()
model.add(LSTM(input_shape=(sequence_length, nb_features), units=64, return_sequences=True, name="lstm_0")) #第一层
model.add(Dropout(0.2, name="dropout_0")) # dropout消除过拟合
model.add(LSTM(units=64, return_sequences=False, name="lstm_1"))
model.add(Dropout(0.2, name="dropout_1"))
model.add(Dense(units=8))
model.add(Dropout(0.2, name="dropout_2"))
model.add(Dense(units=8))
model.add(Dropout(0.2, name="dropout_3"))
model.add(Dense(units=nb_out))
model.add(Activation("relu", name="activation_0"))
model.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae',score_calc])

print(model.summary())

epochs = 500
batch_size = 512

# fit the network
history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

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
plt.legend(['train', 'test'], loc='upper left')
plt.show()
fig_acc.savefig("output/model_regression_loss.png")


model1 = Sequential()
model1.add(LSTM(input_shape=(sequence_length, nb_features), units=64, return_sequences=True, name="lstm_0")) #第一层
model1.add(Dropout(0.2, name="dropout_0")) # dropout消除过拟合
model1.add(LSTM(units=64, return_sequences=False, name="lstm_1"))
model1.add(Dropout(0.2, name="dropout_1"))
model1.add(Dense(units=8))
model1.add(Dropout(0.2, name="dropout_2"))
model1.add(Dense(units=8))
model1.add(Dropout(0.2, name="dropout_3"))
model1.add(Dense(units=nb_out))
model1.add(Activation("relu", name="activation_0"))
model1.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae',score_calc])

print(model1.summary())

epochs = 500
batch_size = 512

# fit the network
history = model1.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path1, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path1))

model2 = Sequential()
model2.add(LSTM(input_shape=(sequence_length, nb_features), units=64, return_sequences=True, name="lstm_0")) #第一层
model2.add(Dropout(0.2, name="dropout_0")) # dropout消除过拟合
model2.add(LSTM(units=64, return_sequences=False, name="lstm_1"))
model2.add(Dropout(0.2, name="dropout_1"))
model2.add(Dense(units=8))
model2.add(Dropout(0.2, name="dropout_2"))
model2.add(Dense(units=8))
model2.add(Dropout(0.2, name="dropout_3"))
model2.add(Dense(units=nb_out))
model2.add(Activation("relu", name="activation_0"))
model2.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae',score_calc])

print(model2.summary())

epochs = 500
batch_size = 512

# fit the network
history = model2.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path2, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path2))


model3 = Sequential()
model3.add(LSTM(input_shape=(sequence_length, nb_features), units=64, return_sequences=True, name="lstm_0")) #第一层
model3.add(Dropout(0.2, name="dropout_0")) # dropout消除过拟合
model3.add(LSTM(units=64, return_sequences=False, name="lstm_1"))
model3.add(Dropout(0.2, name="dropout_1"))
model3.add(Dense(units=8))
model3.add(Dropout(0.2, name="dropout_2"))
model3.add(Dense(units=8))
model3.add(Dropout(0.2, name="dropout_3"))
model3.add(Dense(units=nb_out))
model3.add(Activation("relu", name="activation_0"))
model3.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae',score_calc])

print(model3.summary())

epochs = 500
batch_size = 512

# fit the network
history = model3.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path3, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path3))


model4 = Sequential()
model4.add(LSTM(input_shape=(sequence_length, nb_features), units=64, return_sequences=True, name="lstm_0")) #第一层
model4.add(Dropout(0.2, name="dropout_0")) # dropout消除过拟合
model4.add(LSTM(units=64, return_sequences=False, name="lstm_1"))
model4.add(Dropout(0.2, name="dropout_1"))
model4.add(Dense(units=8))
model4.add(Dropout(0.2, name="dropout_2"))
model4.add(Dense(units=8))
model4.add(Dropout(0.2, name="dropout_3"))
model4.add(Dense(units=nb_out))
model4.add(Activation("relu", name="activation_0"))
model4.compile(loss=score_calc, optimizer='rmsprop', metrics=[root_mean_squared_error, 'mae',score_calc])

print(model4.summary())

epochs = 500
batch_size = 512

# fit the network
history = model4.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=80,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path4, monitor='val_loss',save_best_only=True, mode='min', verbose=0)]
          )

# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path4))