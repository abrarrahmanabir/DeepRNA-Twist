import tensorflow as tf

def custom_loss_SAINT(y_true, y_pred):
    sine_values = y_true[:, :, ::2]
    mask = tf.cast(tf.not_equal(sine_values, 0), tf.float32)
    mask = tf.repeat(mask, repeats=2, axis=2)
    diff = y_true - y_pred
    squared_diff = tf.square(diff) * mask
    masked_mse = tf.reduce_sum(squared_diff) / tf.reduce_sum(mask)
    return masked_mse
