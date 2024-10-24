from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution1D, Dropout, BatchNormalization, Concatenate, Dense, TimeDistributed, Add, Lambda, Softmax, Permute, Dot, Embedding, Layer, add, RepeatVector,AveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import UpSampling1D
from tensorflow import keras
import keras
import pickle
from tensorflow.keras.models import load_model



pickle_file_path = 'emb.pkl' # LOAD LANGUAGE MODEL ENBEDDINGS
df = pd.read_csv('combined_full.csv')   # LOAD DATASET

with open(pickle_file_path, 'rb') as f:
    sequence_embeds = pickle.load(f)
print(len(sequence_embeds))

for i in range(len(sequence_embeds)):
    sequence_embeds[i] = np.squeeze(sequence_embeds[i], axis=0)


MAXIMUM_LENGTH = 150
NUM_OF_RNAs = 405  
valid_bases = ['A', 'U', 'G', 'C']
df = df[df['Base'].isin(valid_bases)]
df.replace('---', pd.NA, inplace=True)
df.dropna(inplace=True)
max_id = NUM_OF_RNAs  
max_len = MAXIMUM_LENGTH  
n_bases = 1 
n_angles = 9  

sequence_array = np.full((max_id+1, max_len), 'X', dtype=str)  
angle_array = np.full((max_id+1, max_len, n_angles), 0) 
i = 0
for group_id, group_data in df.groupby('id'):
    base_sequence = group_data['Base'].tolist()
    padded_base_sequence = base_sequence[:max_len] + ['X'] * max(0, max_len - len(base_sequence))

    sequence_array[group_id, :len(padded_base_sequence)] = padded_base_sequence
    angles = group_data[['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi','eta','theta']].values
    fixed_angles = np.zeros((MAXIMUM_LENGTH, 9))
    N = angles.shape[0]
    original_rows = angles[-min(N, MAXIMUM_LENGTH):]

    fixed_angles[:original_rows.shape[0]] = original_rows
    angle_array[i]= fixed_angles
    i = i + 1


sequence_array.shape, angle_array.shape
char_to_index = {'G': 0, 'A': 1, 'C': 2, 'U': 3}
one_hot_encoded_array = np.zeros((sequence_array.shape[0], sequence_array.shape[1], 4))

for i in range(sequence_array.shape[0]):
    for j in range(sequence_array.shape[1]):
        char = sequence_array[i, j]
        if char in char_to_index:
            one_hot_encoded_array[i, j, char_to_index[char]] = 1

angle_array_radians = np.radians(angle_array)
output_array_shape = angle_array_radians.shape[:-1] + (angle_array_radians.shape[-1] * 2,)
sine_cosine_array = np.zeros(output_array_shape)
for i in range(angle_array_radians.shape[-1]):
    sine_cosine_array[..., 2*i] = np.sin(angle_array_radians[..., i])
    sine_cosine_array[..., 2*i + 1] = np.cos(angle_array_radians[..., i])

sine_cosine_array.shape, sine_cosine_array[0]  , one_hot_encoded_array.shape, one_hot_encoded_array[0]
sine_cosine_values = sine_cosine_array


sequences = []


for sequence in sequence_array:

    filtered_sequence = ''.join([char for char in sequence if char != 'X'])
    sequences.append(filtered_sequence)

#rinalmo
embeddings = sequence_embeds
dim=1280
modified_embeddings = [arr[1:-1] if arr.shape[0] > 1 else np.empty((0, dim)) for arr in embeddings]
embeddings = modified_embeddings

embeddings = [arr[:MAXIMUM_LENGTH, :] if arr.shape[0] > MAXIMUM_LENGTH else arr for arr in embeddings]
max_rows = 150 
def pad_to_max_rows(arr, max_rows):

    padding_needed = max_rows - arr.shape[0]


    return np.pad(arr, ((0, padding_needed), (0, 0)), 'constant')

padded_embeddings = [pad_to_max_rows(arr, max_rows) for arr in embeddings]
embeddings = padded_embeddings

features_2 = np.array(embeddings)
features_1 =  one_hot_encoded_array
min_size_0 = min(features_1.shape[0], features_2.shape[0])
min_size_1 = min(features_1.shape[1], features_2.shape[1])
features_1_truncated = features_1[:min_size_0, :min_size_1, :]
features_2_truncated = features_2[:min_size_0, :min_size_1, :]

features = np.concatenate((features_1_truncated, features_2_truncated), axis=2)

mean = np.mean(features, axis=(0, 1), keepdims=True)
std = np.std(features, axis=(0, 1), keepdims=True)
features = (features - mean) / std

def sine_cosine_to_angle(sine_cosine):
    predicted_angles = np.empty((sine_cosine.shape[0], sine_cosine.shape[1], sine_cosine.shape[2]//2))

    for i in range(0, sine_cosine.shape[2], 2): 
        sine = sine_cosine[:, :, i]
        cosine = sine_cosine[:, :, i + 1]
        angle = np.arctan2(sine, cosine)
        predicted_angles[:, :, i//2] = np.degrees(angle) 

    return predicted_angles


def periodic_mae_for_angle(true_angles, predicted_angles, angle_index):

    
    true_angle = true_angles[:, :, angle_index]
    predicted_angle = predicted_angles[:, :, angle_index]

    mask = true_angle != 0

    filtered_true_angle = true_angle[mask]
    filtered_predicted_angle = predicted_angle[mask]

    abs_diff = np.abs(filtered_true_angle - filtered_predicted_angle)

    adjusted_diff = np.minimum(abs_diff, 360 - abs_diff)

    mae = np.mean(adjusted_diff)
    return mae


def periodic_mae(true_angles, predicted_angles):

    # print(true_angle.shape , predicted_angles.shape)
     
    mask = true_angles != 0
 
 
    filtered_true_angles = true_angles[mask]
    filtered_predicted_angles = predicted_angles[mask]

   
    abs_diff = np.abs(filtered_true_angles - filtered_predicted_angles)

  
    complementary_diff = 360 - abs_diff

    
    min_diff = np.minimum(abs_diff, complementary_diff)


    mae = np.mean(min_diff)

    return mae



def clone_initializer(initializer):
    """Clones an initializer to ensure a new seed.

    As of tensorflow 2.10, we need to clone user passed initializers when
    invoking them twice to avoid creating the same randomized initialization.
    """
    # If we get a string or dict, just return as we cannot and should not clone.
    if not isinstance(initializer, keras.initializers.Initializer):
        return initializer
    config = initializer.get_config()
    return initializer.__class__.from_config(config)


def _check_masks_shapes(inputs, padding_mask, attention_mask):
    mask = padding_mask
    if hasattr(inputs, "_keras_mask") and mask is None:
        mask = inputs._keras_mask
    if mask is not None:
        if len(mask.shape) != 2:
            raise ValueError(
                "`padding_mask` should have shape "
                "(batch_size, target_length). "
                f"Received shape `{mask.shape}`."
            )
    if attention_mask is not None:
        if len(attention_mask.shape) != 3:
            raise ValueError(
                "`attention_mask` should have shape "
                "(batch_size, target_length, source_length). "
                f"Received shape `{mask.shape}`."
            )

def merge_padding_and_attention_mask(
    inputs,
    padding_mask,
    attention_mask,
):
    """Merge the padding mask with a customized attention mask.

    Args:
        inputs: the input sequence.
        padding_mask: the 1D padding mask, of shape
            [batch_size, sequence_length].
        attention_mask: the 2D customized mask, of shape
            [batch_size, sequence1_length, sequence2_length].

    Return:
        A merged 2D mask or None. If only `padding_mask` is provided, the
        returned mask is padding_mask with one additional axis.
    """
    _check_masks_shapes(inputs, padding_mask, attention_mask)
    mask = padding_mask
    if hasattr(inputs, "_keras_mask"):
        if mask is None:
            # If no padding mask is explicitly provided, we look for padding
            # mask from the input data.
            mask = inputs._keras_mask

    if mask is not None:
        # Add an axis for broadcasting, the attention mask should be 2D
        # (not including the batch axis).
        mask = ops.cast(ops.expand_dims(mask, axis=1), "int32")
    if attention_mask is not None:
        attention_mask = ops.cast(attention_mask, "int32")
        if mask is None:
            return attention_mask
        else:
            return ops.minimum(mask, attention_mask)
    return mask

class TransformerEncoder(keras.layers.Layer):

    def __init__(
        self,
        intermediate_dim,
        num_heads,
        dropout=0,
        activation="relu",
        layer_norm_epsilon=1e-05,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        normalize_first=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = keras.activations.get(activation)
        self.layer_norm_epsilon = layer_norm_epsilon
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.normalize_first = normalize_first
        self.supports_masking = True

    def build(self, inputs_shape):
        # Infer the dimension of our hidden feature size from the build shape.
        hidden_dim = inputs_shape[-1]
        # Attention head size is `hidden_dim` over the number of heads.
        key_dim = int(hidden_dim // self.num_heads)
        if key_dim == 0:
            raise ValueError(
                "Attention `key_dim` computed cannot be zero. "
                f"The `hidden_dim` value of {hidden_dim} has to be equal to "
                f"or greater than `num_heads` value of {self.num_heads}."
            )

        # Self attention layers.
        self._self_attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=key_dim,
            dropout=self.dropout,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="self_attention_layer",
        )
        if hasattr(self._self_attention_layer, "_build_from_signature"):
            self._self_attention_layer._build_from_signature(
                query=inputs_shape,
                value=inputs_shape,
            )
        else:
            self._self_attention_layer.build(
                query_shape=inputs_shape,
                value_shape=inputs_shape,
            )
        self._self_attention_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="self_attention_layer_norm",
        )
        self._self_attention_layer_norm.build(inputs_shape)
        self._self_attention_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="self_attention_dropout",
        )

        # Feedforward layers.
        self._feedforward_layer_norm = keras.layers.LayerNormalization(
            epsilon=self.layer_norm_epsilon,
            dtype=self.dtype_policy,
            name="feedforward_layer_norm",
        )
        self._feedforward_layer_norm.build(inputs_shape)
        self._feedforward_intermediate_dense = keras.layers.Dense(
            self.intermediate_dim,
            activation=self.activation,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_intermediate_dense",
        )
        self._feedforward_intermediate_dense.build(inputs_shape)
        self._feedforward_output_dense = keras.layers.Dense(
            hidden_dim,
            kernel_initializer=clone_initializer(self.kernel_initializer),
            bias_initializer=clone_initializer(self.bias_initializer),
            dtype=self.dtype_policy,
            name="feedforward_output_dense",
        )
        intermediate_shape = list(inputs_shape)
        intermediate_shape[-1] = self.intermediate_dim
        self._feedforward_output_dense.build(tuple(intermediate_shape))
        self._feedforward_dropout = keras.layers.Dropout(
            rate=self.dropout,
            dtype=self.dtype_policy,
            name="feedforward_dropout",
        )
        self.built = True

    def call(self, inputs, padding_mask=None, attention_mask=None):
        """Forward pass of the TransformerEncoder.

        Args:
            inputs: a Tensor. The input data to TransformerEncoder, should be
                of shape [batch_size, sequence_length, hidden_dim].
            padding_mask: a boolean Tensor. It indicates if the token should be
                masked because the token is introduced due to padding.
                `padding_mask` should have shape [batch_size, sequence_length].
            attention_mask: a boolean Tensor. Customized mask used to mask out
                certain tokens. `attention_mask` should have shape
                [batch_size, sequence_length, sequence_length].

        Returns:
            A Tensor of the same shape as the `inputs`.
        """
        x = inputs  # Intermediate result.

        # Compute self attention mask.
        self_attention_mask = merge_padding_and_attention_mask(
            inputs, padding_mask, attention_mask
        )

        # Self attention block.
        residual = x
        if self.normalize_first:
            x = self._self_attention_layer_norm(x)
        x = self._self_attention_layer(
            query=x,
            value=x,
            attention_mask=self_attention_mask,
        )
        x = self._self_attention_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._self_attention_layer_norm(x)

        # Feedforward block.
        residual = x
        if self.normalize_first:
            x = self._feedforward_layer_norm(x)
        x = self._feedforward_intermediate_dense(x)
        x = self._feedforward_output_dense(x)
        x = self._feedforward_dropout(x)
        x = x + residual
        if not self.normalize_first:
            x = self._feedforward_layer_norm(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "intermediate_dim": self.intermediate_dim,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "activation": keras.activations.serialize(self.activation),
                "layer_norm_epsilon": self.layer_norm_epsilon,
                "kernel_initializer": keras.initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": keras.initializers.serialize(
                    self.bias_initializer
                ),
                "normalize_first": self.normalize_first,
            }
        )
        return config

    def compute_output_shape(self, inputs_shape):
        return inputs_shape
    

def inceptionBlock(x):
    _drop_rate_ = 0.1
    x = BatchNormalization()(x)

    conv1_1 = Convolution1D(filters=100, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv1_1 = Dropout(_drop_rate_)(conv1_1)
    conv1_1 = BatchNormalization()(conv1_1)

    conv2_1 = Convolution1D(filters=100, kernel_size=1, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv2_1 = Dropout(_drop_rate_)(conv2_1)
    conv2_1 = BatchNormalization()(conv2_1)
    conv2_2 = Convolution1D(filters=200, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv2_1)
    conv2_2 = Dropout(_drop_rate_)(conv2_2)
    conv2_2 = BatchNormalization()(conv2_2)

    conv3_1 = Convolution1D(filters=100, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv3_1 = Dropout(_drop_rate_)(conv3_1)
    conv3_1 = BatchNormalization()(conv3_1)
    conv3_2 = Convolution1D(filters=200, kernel_size=5, activation='relu', padding='same', kernel_regularizer=l2(0.001))(conv3_1)
    conv3_2 = Dropout(_drop_rate_)(conv3_2)
    conv3_2 = BatchNormalization()(conv3_2)

    conv4_1 = Convolution1D(filters=100, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(x)
    conv4_1 = Dropout(_drop_rate_)(conv4_1)
    conv4_1 = BatchNormalization()(conv4_1)
    conv4_2 = Convolution1D(filters=200, kernel_size=3, activation='relu', dilation_rate=2, padding='same', kernel_regularizer=l2(0.001))(conv4_1)
    conv4_2 = Dropout(_drop_rate_)(conv4_2)
    conv4_2 = BatchNormalization()(conv4_2)

    concat = Concatenate()([conv1_1, conv2_2, conv3_2, conv4_2])
    concat = BatchNormalization()(concat)

    return concat



def shape_list(x):
    if backend.backend() != 'theano':
        tmp = backend.int_shape(x)
    else:
        tmp = x.shape
    tmp = list(tmp)
    tmp[0] = -1
    return tmp



def attention_scaled_dot(activations, attention_mask, num_heads=4):
    d_model = int(activations.shape[2])
    _drop_rate_ = .1
    #print(d_model)
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    depth = d_model // num_heads

    def split_heads(x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    batch_size = tf.shape(activations)[0]

    # Linear layers
    WQ = TimeDistributed(Dense(d_model, use_bias=False))(activations)
    WK = TimeDistributed(Dense(d_model, use_bias=False))(activations)
    WV = TimeDistributed(Dense(d_model, use_bias=False))(activations)

    # Split heads
    Q = split_heads(WQ, batch_size)
    K = split_heads(WK, batch_size)
    V = split_heads(WV, batch_size)

    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    attention_mask_ = attention_mask[:, tf.newaxis, tf.newaxis, :]
    add_attention_mask = scaled_attention_logits + (attention_mask_ * -1e9)

    attention_weights = tf.nn.softmax(add_attention_mask, axis=-1)
    attention_weights = Dropout(_drop_rate_)(attention_weights)

    output = tf.matmul(attention_weights, V)


    output = tf.transpose(output, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(output, (batch_size, -1, d_model))


    attention_output = TimeDistributed(Dense(d_model))(concat_attention)

    return attention_output


from tensorflow.keras.layers import Layer
class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._x = self.add_weight(name='alpha', shape=(), initializer=tf.constant_initializer(0.2), trainable=True, dtype='float32')
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        A, B = inputs
        result = add([self._x*A, (1-self._x)*B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]



def _get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2. * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb) for pos in range(max_len)
    ], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc

def position_embedding(pos_ids, output_dim=50):
    max_len = MAXIMUM_LENGTH
    output_dim = int(output_dim)
    pos_emb = Embedding(
        max_len,
        output_dim,
        trainable=False,
        weights=[_get_pos_encoding_matrix(max_len, output_dim)]
    )(pos_ids)
    pos_emb = Dropout(.1)(pos_emb)
    pos_emb = BatchNormalization()(pos_emb)
    return pos_emb

def attention_module(x, attention_mask, pos_ids=None, drop_rate=.1):
    # original_dim = int(x.shape[-1])
    # if pos_ids is not None:
    #     pos_embedding = position_embedding(pos_ids=pos_ids, output_dim=original_dim)
    #     x = Add()([x, pos_embedding])

    # att_layer = attention_scaled_dot(x, attention_mask)
    # att_layer = Dropout(drop_rate)(att_layer)
    # x = MyLayer()([att_layer, x])
    # x = Dropout(drop_rate)(x)
    # x = BatchNormalization()(x)
    return x

def dilated_cnn_block(x, filters=100, kernel_size=3, dilation_rate=2):
    x = Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(x)
    x = Dropout(0.1)(x)
    x = BatchNormalization()(x)
    return x


def deep3iBLock_with_attention(x, attention_mask, pos_ids=None):
    block1_1 = inceptionBlock(x)
    block1_1_dilated = dilated_cnn_block(block1_1)
    block1_1_attention = attention_module(block1_1_dilated, attention_mask, pos_ids)

    block2_1 = inceptionBlock(block1_1_attention)
    block2_1_dilated = dilated_cnn_block(block2_1)
    block2_1_attention = attention_module(block2_1_dilated, attention_mask, pos_ids)

    block3_1_pool = AveragePooling1D(pool_size=2, padding='same')(x)
    block3_1_conv = Convolution1D(filters=100, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block3_1_pool)
    block3_1_conv = BatchNormalization()(block3_1_conv)

    block3_1_upsampled = UpSampling1D(size=2)(block3_1_conv)
    block3_1_attention = attention_module(block3_1_upsampled, attention_mask, pos_ids)

    concat = Concatenate()([block1_1_attention, block2_1_attention, block3_1_attention])
    concat = BatchNormalization()(concat)

    return concat


def get_model(num_features):
    _drop_rate_ = 0.4
    d_model = 64  # Set the dimensionality of the model

    # Inputs
    main_input = Input(shape=(MAXIMUM_LENGTH, num_features,), name='main_input')
    attention_mask = Input(shape=(MAXIMUM_LENGTH,), name='attention_mask')
    pos_ids = Input(batch_shape=(None, MAXIMUM_LENGTH), name='position_input', dtype='int32')


    transformer_encoder_1 = TransformerEncoder(
        intermediate_dim=256,  # The dimensionality of the feed-forward network
        num_heads=4,           # Number of attention heads
        dropout=_drop_rate_,   # Dropout rate

    )
    transformer_output_1 = transformer_encoder_1(main_input)


    block1 = deep3iBLock_with_attention(transformer_output_1, attention_mask, pos_ids)
    block2 = deep3iBLock_with_attention(block1, attention_mask, pos_ids)

    output_2a3i_attention = attention_module(block2, attention_mask, pos_ids)



   
    conv11 = Convolution1D(
        filters=d_model,
        kernel_size=11,
        activation='relu',
        padding='same',
        kernel_regularizer=l2(0.001)
    )(output_2a3i_attention) # transformer_output

    conv11_attention = attention_module(conv11, attention_mask, pos_ids)
    dense1 = TimeDistributed(Dense(units=256, activation='relu'))(conv11_attention)
    dense1 = Dropout(_drop_rate_)(dense1)
    dense1_attention = attention_module(dense1, attention_mask, pos_ids)

    # Output layer
    main_output = TimeDistributed(Dense(units=18, activation='tanh', name='main_output'))(dense1_attention)
    model = Model(inputs=[main_input, attention_mask, pos_ids], outputs=main_output)

    return model



def custom_loss_SAINT(y_true, y_pred):

    sine_values = y_true[:, :, ::2]  
    mask = tf.cast(tf.not_equal(sine_values, 0), tf.float32)  

    mask = tf.repeat(mask, repeats=2, axis=2)  


    diff = y_true - y_pred
    squared_diff = tf.square(diff) * mask  

    masked_mse = tf.reduce_sum(squared_diff) / tf.reduce_sum(mask) 

    return masked_mse


num_sequences = 404
sequence_length = 150
sine_cosine_values = sine_cosine_values[:404, :, :]
model_path = 'trained_DeepRNAtwist.h5'
attention_masks = np.ones((num_sequences, sequence_length))
pos_ids = np.tile(np.arange(sequence_length), (num_sequences, 1))
model_2 = load_model(model_path, custom_objects={'custom_loss_SAINT': custom_loss_SAINT , 'TransformerEncoder' : TransformerEncoder}) # correct
predictions = model_2.predict([features, attention_masks, pos_ids])
predicted_angles_degrees = sine_cosine_to_angle(predictions)
true_angles_degrees = sine_cosine_to_angle(sine_cosine_values)


for i in range(9):
  angle_index = i  
  mae_for_angle = periodic_mae_for_angle(true_angles_degrees, predicted_angles_degrees, angle_index)

  print(f"Mean Absolute Error for Angle {angle_index + 1}: {mae_for_angle} degrees")


