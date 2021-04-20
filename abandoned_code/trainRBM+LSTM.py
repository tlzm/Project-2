# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 08:27:03 2020

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
from keras.layers import Flatten,RNN
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.engine.topology import Layer #这里的Layer是一个父类，下面的MyLayer将会继承Layer
from RBM import RBMhidden,RBMvisible
from keras.optimizers import Adam



output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/train.csv')
n_turb = train_data['id'].unique().max()

# pick a large window size of 30 cycles
sequence_length = 30

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
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]

def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)
 
#def score_calc(y_true, y_pred):
 #   return K.mean(((-1+K.exp((K.abs(y_pred-y_true)+y_pred-y_true)/20))+(-1+K.exp((K.abs(y_true-y_pred)+y_true-y_pred)/26))))       
        
# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols) # 为啥又建立一遍

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
    return data_matrix[seq_length: num_elements, :]

# generate labels
label_gen = [reshapeLabel(train_data[train_data['id']==id]) for id in range(1, n_turb + 1)]

label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)



# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) # 均方根差

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]

model = Sequential()

model.add(RBMhidden(input_shape=(sequence_length, nb_features),units=64,return_sequences=True,activation='relu',kernel_regularizer=keras.regularizers.l2(0.000001),
                    activity_regularizer=keras.regularizers.l2(0.000001)))
model.add(RBMvisible(units=64,return_sequences=True,activation='relu',kernel_regularizer=keras.regularizers.l2(0.000001),
                     activity_regularizer=keras.regularizers.l2(0.000001)))
#model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(units=32, return_sequences=True, name="lstm_1",activation='sigmoid'))
model.add(Dropout(0.1, name="dropout_1"))
model.add(LSTM(units=32, return_sequences=False, name="lstm_2",activation='sigmoid'))
model.add(Dropout(0.1, name="dropout_2"))
model.add(Dense(units=8,activation='sigmoid'))
model.add(Dropout(0.1, name="dropout_3"))
model.add(Dense(units=nb_out, name="dense_0"))
#model.add(Activation("relu", name="activation_0"))


model.compile(loss='mse', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0), metrics=[root_mean_squared_error,'mae', score_calc])

print(model.summary())

epochs = 5000
batch_size = 500

# fit the network
history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_split=0.05, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100,verbose=0, mode='min'),
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
