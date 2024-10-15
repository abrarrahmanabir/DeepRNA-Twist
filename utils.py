import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Add, Dropout, Dense, Concatenate, Convolution1D, TimeDistributed

def attention_module(x, attention_mask, pos_ids=None, drop_rate=.1):
    original_dim = int(x.shape[-1])
    att_layer = attention_scaled_dot(x, attention_mask)
    att_layer = Dropout(drop_rate)(att_layer)
    x = MyLayer()([att_layer, x])
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    return x

def inceptionBlock(x):
    _drop_rate_ = 0.1
    x = BatchNormalization()(x)
    conv1_1 = Convolution1D(filters=100, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv1_1 = Dropout(_drop_rate_)(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)
    conv2_1 = Convolution1D(filters=100, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv2_2 = Convolution1D(filters=200, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
    concat = Concatenate()([conv1_1, conv2_2])
    concat = BatchNormalization()(concat)
    return concat
