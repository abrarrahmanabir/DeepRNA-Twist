import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Add, Dropout, Dense, Concatenate, Convolution1D, TimeDistributed, Embedding
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2

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


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self._x = self.add_weight(name='alpha', shape=(), initializer=tf.constant_initializer(0.2), trainable=True, dtype='float32')
        super(MyLayer, self).build(input_shape)

    def call(self, inputs):
        A, B = inputs
        result = Add()([self._x * A, (1 - self._x) * B])
        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]


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


def attention_scaled_dot(activations, attention_mask, num_heads=4):
    d_model = int(activations.shape[2])
    _drop_rate_ = .1
    depth = d_model // num_heads

    def split_heads(x, batch_size):
        x = tf.reshape(x, (batch_size, -1, num_heads, depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    batch_size = tf.shape(activations)[0]
    WQ = TimeDistributed(Dense(d_model, use_bias=False))(activations)
    WK = TimeDistributed(Dense(d_model, use_bias=False))(activations)
    WV = TimeDistributed(Dense(d_model, use_bias=False))(activations)

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


def attention_module(x, attention_mask, pos_ids=None, drop_rate=.1):
    original_dim = int(x.shape[-1])
    if pos_ids is not None:
        pos_embedding = position_embedding(pos_ids=pos_ids, output_dim=original_dim)
        x = Add()([x, pos_embedding])

    att_layer = attention_scaled_dot(x, attention_mask)
    att_layer = Dropout(drop_rate)(att_layer)
    x = MyLayer()([att_layer, x])
    x = Dropout(drop_rate)(x)
    x = BatchNormalization()(x)
    return x


def _get_pos_encoding_matrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2. * (j // 2) / d_emb) for j in range(d_emb)]
        if pos != 0 else np.zeros(d_emb) for pos in range(max_len)
    ], dtype=np.float32)
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])
    return pos_enc


def position_embedding(pos_ids, output_dim=50):
    max_len = pos_ids.shape[1]
    output_dim = int(output_dim)
    pos_emb = Embedding(max_len, output_dim, trainable=False, weights=[_get_pos_encoding_matrix(max_len, output_dim)])(pos_ids)
    pos_emb = Dropout(.1)(pos_emb)
    pos_emb = BatchNormalization()(pos_emb)
    return pos_emb


def dilated_cnn_block(x, filters=100, kernel_size=3, dilation_rate=2):
    x = Convolution1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu', dilation_rate=dilation_rate)(x)
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

    block3_1_pool = tf.keras.layers.AveragePooling1D(pool_size=2, padding='same')(x)
    block3_1_conv = Convolution1D(filters=100, kernel_size=3, activation='relu', padding='same', kernel_regularizer=l2(0.001))(block3_1_pool)
    block3_1_conv = BatchNormalization()(block3_1_conv)
    block3_1_upsampled = tf.keras.layers.UpSampling1D(size=2)(block3_1_conv)
    block3_1_attention = attention_module(block3_1_upsampled, attention_mask, pos_ids)

    concat = Concatenate()([block1_1_attention, block2_1_attention, block3_1_attention])
    concat = BatchNormalization()(concat)

    return concat

