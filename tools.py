import numpy as np
import tensorflow as tf
import random as python_random

###############
### random seeds
###############

def set_seed():
    np.random.seed(100)
    python_random.seed(100)
    tf.random.set_seed(100)
    
OVERLAY_LABELS = tf.reshape(tf.one_hot(tf.range(10), depth=28*28), (10, 28, 28))
OVERLAY_MASK = tf.expand_dims(1. - tf.reduce_sum(OVERLAY_LABELS, axis=0), 0)

###############
### Data processing
###############

@tf.function
def preprocess(X, y):
    return (tf.cast(X, tf.float32)/255., tf.cast(y, tf.int32))

@tf.function
def _overlay_y_in_X(X, y):
    X = X * OVERLAY_MASK + tf.gather(OVERLAY_LABELS, y)
    return X

@tf.function
def overlay_y_in_X(X_pos, X_neg, y_pos, y_neg):
    y_neg_rand = tf.random.uniform(tf.shape(y_neg), maxval=10, dtype=tf.int32)
    X_pos = _overlay_y_in_X(X_pos, y_pos)
    X_neg = _overlay_y_in_X(X_neg, y_neg_rand)
    return X_pos, X_neg, y_pos, y_neg

@tf.function
def overlay_each_label_in_one_X(X):
    m = tf.shape(X)[0]
    X = tf.tile(X, [10, 1, 1, ])
    y = tf.repeat(tf.range(10), [m]*10)
    return _overlay_y_in_X(X, y)