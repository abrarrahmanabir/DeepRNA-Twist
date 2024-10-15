import tensorflow as tf
from tensorflow.keras.layers import Convolution1D, Dense, Dropout, BatchNormalization, TimeDistributed, Concatenate, Input, UpSampling1D, AveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from utils import attention_module, inceptionBlock, TransformerEncoder, deep3iBLock_with_attention

def get_model(num_features, MAXIMUM_LENGTH):
    _drop_rate_ = 0.4
    d_model = 64

    main_input = Input(shape=(MAXIMUM_LENGTH, num_features,), name='main_input')
    attention_mask = Input(shape=(MAXIMUM_LENGTH,), name='attention_mask')
    pos_ids = Input(batch_shape=(None, MAXIMUM_LENGTH), name='position_input', dtype='int32')

    transformer_encoder_1 = TransformerEncoder(intermediate_dim=256, num_heads=4, dropout=_drop_rate_)
    transformer_output_1 = transformer_encoder_1(main_input)

    block1 = deep3iBLock_with_attention(transformer_output_1, attention_mask, pos_ids)
    block2 = deep3iBLock_with_attention(block1, attention_mask, pos_ids)

    output_2a3i_attention = attention_module(block2, attention_mask, pos_ids)

    conv11 = Convolution1D(filters=d_model, kernel_size=11, activation='relu', padding='same', kernel_regularizer=l2(0.001))(output_2a3i_attention)
    conv11_attention = attention_module(conv11, attention_mask, pos_ids)
    dense1 = TimeDistributed(Dense(units=256, activation='relu'))(conv11_attention)
    dense1 = Dropout(_drop_rate_)(dense1)
    dense1_attention = attention_module(dense1, attention_mask, pos_ids)

    main_output = TimeDistributed(Dense(units=18, activation='tanh', name='main_output'))(dense1_attention)
    model = Model(inputs=[main_input, attention_mask, pos_ids], outputs=main_output)
    
    return model
