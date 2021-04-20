# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 07:58:22 2020

@author: tianc
"""
# -*- coding: utf-8 -*-


import tensorflow as tf
import random
import keras
import keras.backend as K
from keras.layers.core import Activation
from keras.models import Sequential
from keras.layers import Dense, Dropout,SeparableConv1D, LSTM,Lambda,Flatten,Bidirectional,Conv1D,MaxPooling1D,BatchNormalization,Permute,Reshape,RepeatVector,multiply
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras import objectives
from keras import regularizers
from keras.models import load_model
from keras.layers.wrappers import Bidirectional, TimeDistributed

input_path ='model/pretrain_model_v0.h5'
output_path = 'model/regression_model_v0.h5'
train_data = pd.read_csv('input/train.csv')
test_data = pd.read_csv("input/test.csv")
n_turb = train_data['id'].unique().max()

output_path1 = 'model/regression_model_v1.h5'
output_path2 = 'model/regression_model_v2.h5'
output_path3 = 'model/regression_model_v3.h5'
output_path4 = 'model/regression_model_v4.h5'
output_path5 = 'model/regression_model_v5.h5'



# pick a large window size of 30 cycles
sequence_length = 31

SINGLE_ATTENTION_VECTOR = True


# pick the feature columns 
#sensor_cols = ['s' + str(i) for i in range(1,22)]
#sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
#sequence_cols.extend(sensor_cols) 
sequence_cols = ['s2', 's3','s4', 's7', 's8',
         's9', 's11', 's12', 's13', 's14', 's15', 's17', 's20', 's21']
#2, 3, 4, 7, 8, 9,11, 12, 13, 14, 15, 17, 20 and 21


seq_array_test_last = [test_data[test_data['id']==id][sequence_cols].values[-sequence_length:] 
                       for id in range(1, n_turb + 1) if len(test_data[test_data['id']==id]) >= sequence_length]
xtest = np.asarray(seq_array_test_last).astype(np.float32)


y_mask = [len(test_data[test_data['id']==id]) >= sequence_length for id in test_data['id'].unique()]
label_array_test_last = test_data.groupby('id')['RUL'].nth(-1)[y_mask].values

ytest = label_array_test_last.reshape(label_array_test_last.shape[0],1).astype(np.float32)

def attention_block(inputs, TIME_STEPS):

    input_dim = int(inputs.shape[2])  # input_dim = 100

    a = Permute((2, 1))(inputs)  # a.shape = (?, 100, ?)

    a = Reshape((input_dim, TIME_STEPS))(a)  # a.shape = (?, 100, 30)this line is not useful. It's just to know which dimension is what.

    a = Dense(TIME_STEPS, activation='softmax')(a)  # a.shape = (?, 100, 30)

    if SINGLE_ATTENTION_VECTOR:

        a = Lambda(lambda x: K.mean(x, axis=1))(a)  # a.shape = (?, 30)

        a = RepeatVector(input_dim)(a)  # a.shape = (?, 100, 30) RepeatVector层将输入重复n次

    a_probs = Permute((2, 1))(a)  # a.shape = (?, 30, 100)

    # output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')

    output_attention_mul = multiply([inputs, a_probs])  # [?, 30, 100]

    return output_attention_mul



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

# random for selecting data
idxs_annot = range(feat_array.shape[0])
random.seed(0)
idxs_annot = np.random.choice(feat_array.shape[0],15867)


#unlabel
x_train_unlabeled = feat_array
x_train_labeled   = feat_array[idxs_annot]
y_train_labeled   = label_array[idxs_annot]

n_rep = x_train_unlabeled.shape[0] // x_train_labeled.shape[0]
x_train_labeled_rep = np.concatenate([x_train_labeled]*n_rep)
y_train_labeled_rep = np.concatenate([y_train_labeled]*n_rep)


# MODEL

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) # 均方根差

def exps(y_true, y_pred):
        return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)+K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) #


def score_calc(y_true, y_pred):
    return K.mean((-1+K.exp(K.relu(y_true-y_pred)/13)+-1+K.exp(K.relu(y_pred-y_true)/10)),axis=-1)

nb_features = feat_array.shape[2]
nb_out = label_array.shape[1]


encoder = load_model(input_path)

x = keras.Input(shape=(sequence_length, 7))

x01=LSTM(units=256, return_sequences=True)(x)
x02=Dropout(0.2)(x01)
x03=LSTM(units=64, return_sequences=True)(x02)
x04=Dropout(0.2)(x03)
x05=LSTM(units=16, return_sequences=False)(x04)
x06=Dropout(0.2)(x05)
x07=Dense(units=16, activation='swish')(x06)

# x10=Permute((2, 1))(x)
# x11=attention_block(x10, 7)
# x12=Permute((2, 1))(x11)

# x13=Conv1D(filters=64, kernel_size=2, activation='swish')(x11)
# x14=Conv1D(filters=32, kernel_size=2, activation='swish')(x13)
# x15=Conv1D(filters=16, kernel_size=2,activation='swish')(x13)
# x16=Flatten()(x15)
# x17=Dense(units=16, activation='swish')(x16)

# x18=Dense(units=64, activation='swish')(x16)
# x19=Dense(units=16, activation='swish')(x18)



x11=attention_block(x, 31)
x12=Permute((2, 1))(x11)
x13=Reshape((7,31,1))(x11)
x14=TimeDistributed(Conv1D(filters=64, kernel_size=2,strides=1,padding='same',activation='swish',dilation_rate=1))(x13)
x15=TimeDistributed(Conv1D(filters=64, kernel_size=2,strides=1, padding='same',activation='swish',dilation_rate=1))(x14)
x16=TimeDistributed(SeparableConv1D(filters=1, kernel_size=1,padding='same', activation='swish'))(x15)
z=keras.layers.add([x13, x16])
x17=Reshape((7,31))(x16)

x18=attention_block(x17, 7)
x19=Permute((1,2))(x18)

x20=Flatten()(x19)
x21=Dense(units=64, activation='swish')(x20)
x22=Dense(units=16, activation='swish')(x21)


y=keras.layers.Concatenate(axis= 1)([x07, x22])
# z=keras.layers.Concatenate(axis= 1)([x07, x19])


fix = keras.Model(x, y)

# print(fix.summary())

#Bidirectional
model = Sequential()
model.add(encoder)
model.add(fix)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model.add(Dropout(0.2, name="dropout_4"))
model.add(Dense(units=nb_out))
model.add(Activation("swish", name="activation_0"))
model.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model.summary())

epochs = 400
batch_size = 256

# fit the network
history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=200,verbose=0, mode='min'),
                        keras.callbacks.ModelCheckpoint(output_path, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
# history = model.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, validation_data=[xtest,ytest], verbose=1,
#           callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=70,verbose=0, mode='min'),
#                        keras.callbacks.ModelCheckpoint(output_path, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
#           )
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




fix1 = keras.Model(x, y)

#Bidirectional
model1 = Sequential()
model1.add(encoder)
model1.add(fix1)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model1.add(Dropout(0.2, name="dropout_4"))
model1.add(Dense(units=nb_out))
model1.add(Activation("swish", name="activation_0"))
model1.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model1.summary())

epochs = 400
batch_size = 256

# fit the network
history = model1.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=220,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path1, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path1))


fix2 = keras.Model(x, y)

#Bidirectional
model2 = Sequential()
model2.add(encoder)
model2.add(fix2)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model2.add(Dropout(0.2, name="dropout_4"))
model2.add(Dense(units=nb_out))
model2.add(Activation("swish", name="activation_0"))
model2.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model2.summary())

epochs = 400
batch_size = 256

# fit the network
history = model2.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=180,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path2, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path2))



fix3 = keras.Model(x, y)

#Bidirectional
model3 = Sequential()
model3.add(encoder)
model3.add(fix3)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model3.add(Dropout(0.2, name="dropout_4"))
model3.add(Dense(units=nb_out))
model3.add(Activation("swish", name="activation_0"))
model3.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model3.summary())

epochs = 400
batch_size = 256

# fit the network
history = model3.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=180,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path3, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path3))



fix4 = keras.Model(x, y)

#Bidirectional
model4 = Sequential()
model4.add(encoder)
model4.add(fix4)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model4.add(Dropout(0.2, name="dropout_4"))
model4.add(Dense(units=nb_out))
model4.add(Activation("swish", name="activation_0"))
model4.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model4.summary())

epochs = 400
batch_size = 256

# fit the network
history = model4.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=180,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path4, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path4))


fix5 = keras.Model(x,y)

#Bidirectional
model5 = Sequential()
model5.add(encoder)
model5.add(fix5)
#model.add(Dropout(0.2, name="dropout_3"))
#model.add(Dense(units=16))
model5.add(Dropout(0.2, name="dropout_4"))
model5.add(Dense(units=nb_out))
model5.add(Activation("swish", name="activation_0"))
model5.compile(loss=score_calc, optimizer=Adam(lr=1e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0005), metrics=[root_mean_squared_error, exps,'mae', score_calc])
#model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[root_mean_squared_error, exps,'mae', score_calc])
print(model5.summary())

epochs = 400
batch_size = 256

# fit the network
history = model5.fit(feat_array, label_array, epochs=epochs, batch_size=batch_size, shuffle=True,validation_split=0.1, verbose=1,
          callbacks = [keras.callbacks.EarlyStopping(monitor='val_score_calc', min_delta=0, patience=180,verbose=0, mode='min'),
                       keras.callbacks.ModelCheckpoint(output_path5, monitor='val_score_calc',save_best_only=True, mode='min', verbose=0)]
          )
#validation_data=(test_array,tlabel_array)
# list all data in history
print(history.history.keys())
print("Model saved as {}".format(output_path5))