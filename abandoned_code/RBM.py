# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 10:55:00 2020

@author: 24687
"""






from keras import backend as K

from keras import activations

from keras import initializers

from keras import regularizers

from keras import constraints

from keras.engine.base_layer import InputSpec

from keras.engine.base_layer import Layer
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from keras.utils import conv_utils

from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import standard_ops


import tensorflow as tf
from tensorflow.python.keras import layers, Model, backend, losses
from enum import Enum




class RBMhidden(Layer):
    
    
    def __init__(self,
                 units=14,
                 name='rbmhidden',
                 
                 return_sequences=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
            
        super(RBMhidden, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        
 
        self.units = int(units) if not isinstance(units, int) else units
        self.return_sequences = return_sequences
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
  
            
            

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1]) 


        self.h = self.add_weight(shape=(last_dim,self.units),
                                 initializer='zeros',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name="h")        

        self.built = True  # 最后这句话一定要加上            


    
    def call(self, inputs):
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            h_prob = standard_ops.sigmoid(standard_ops.tensordot(inputs, self.h,[[rank - 1], [0]]))
            h_state = tf.nn.relu(tf.sign(h_prob - backend.random_uniform(tf.shape(h_prob))))
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                 h_prob = sparse_ops.sigmoid(sparse_ops.sparse_tensor_dense_matmul(inputs, self.h))
                 h_state = tf.nn.relu(tf.sign(h_prob - backend.random_uniform(tf.shape(h_prob))))
            else:
                 h_prob = gen_math_ops.sigmoid(gen_math_ops.mat_mul(inputs, self.h))
                 h_state = tf.nn.relu(tf.sign(h_prob - backend.random_uniform(tf.shape(h_prob))))
        

        return h_state
    
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    
    
    def get_config(self):
        config = {
            'units': self.units,
            'return_sequences':self.return_sequences,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(RBMhidden, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
   
class RBMvisible(Layer):
    
    
    def __init__(self,
                 units=32,
                 name='rbmvisible',
                 
                 return_sequences=False,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        
            
        super(RBMvisible, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        
 
        self.units = int(units) if not isinstance(units, int) else units
        self.return_sequences = return_sequences
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True
  
            
            

    def build(self, input_shape):
        dtype = dtypes.as_dtype(self.dtype or K.floatx())
        input_shape = tensor_shape.TensorShape(input_shape)
        last_dim = tensor_shape.dimension_value(input_shape[-1]) 


        self.v = self.add_weight(shape=(last_dim,self.units),
                                 initializer='zeros',
                                 regularizer=self.kernel_regularizer,
                                 trainable=True,
                                 name="h")        

        self.built = True  # 最后这句话一定要加上            


    
    def call(self, inputs):
        rank = inputs.shape.rank
        if rank is not None and rank > 2:
            v_prob = standard_ops.sigmoid(standard_ops.tensordot(inputs, self.v,[[rank - 1], [0]]))
            v_state = tf.nn.relu(tf.sign(v_prob - backend.random_uniform(tf.shape(v_prob))))
        else:
            inputs = math_ops.cast(inputs, self._compute_dtype)
            if K.is_sparse(inputs):
                 v_prob = sparse_ops.sigmoid(sparse_ops.sparse_tensor_dense_matmul(inputs, self.v))
                 v_state = tf.nn.relu(tf.sign(v_prob - backend.random_uniform(tf.shape(v_prob))))
            else:
                 v_prob = gen_math_ops.sigmoid(gen_math_ops.mat_mul(inputs, self.v))
                 v_state = tf.nn.relu(tf.sign(v_prob - backend.random_uniform(tf.shape(v_prob))))
        

        return v_state
    
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
    
    
    
    def get_config(self):
        config = {
            'units': self.units,
            'return_sequences':self.return_sequences,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(RBMvisible, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))














