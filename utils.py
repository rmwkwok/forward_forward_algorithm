import numpy as np
import tensorflow as tf
import random as python_random

def set_seed(seed=100):
    np.random.seed(seed)
    python_random.seed(seed)
    tf.random.set_seed(seed)
    
# @tf.function
# def flatten(X):
#     shape = tf.shape(X)
#     return tf.reshape(X, [shape[0], shape[1], shape[2]*shape[3]*shape[4]])

@tf.function
def preNorm(X):
    '''
    Get the L2 norm over all axes but the zeroth, and normalize `X`
    with that norm.
    '''
    axis = tf.range(tf.rank(X))[1:]
    norm = tf.math.sqrt(tf.reduce_sum(X**2, axis=axis, keepdims=True))
    return X/(norm + 1e-7)

@tf.function
def data_preprocess(X, y):
    '''
    Normalize `X` and do type-casting.
    '''
    return (tf.cast(X, tf.float32)/255., tf.cast(y, tf.int32))
    
###############
### Overlay labels on X
###############

NUM_CLASS = 5 # Only digits 0 to 4 are positive samples
IMG_SHAPE = (28, 28, )
IMG_SIZE = tf.reduce_prod(IMG_SHAPE)

# OVERLAY_LABELS: shape (5, 28, 28)
OVERLAY_LABELS = tf.reshape(
    tf.one_hot(tf.range(NUM_CLASS), depth=IMG_SIZE), 
    (NUM_CLASS, ) + IMG_SHAPE
)

OVERLAY_LABELS_SUM = tf.reduce_sum(OVERLAY_LABELS, keepdims=True, axis=0)

# OVERLAY_DEFAULT: shape (1, 28, 28)
# The first 5 elements are 0.2, rest are 0
OVERLAY_DEFAULT = OVERLAY_LABELS_SUM/NUM_CLASS

# OVERLAY_BITMASK: shape (1, 28, 28)
# The first 5 elements are 0, rest are 1
OVERLAY_BITMASK = 1. - OVERLAY_LABELS_SUM

@tf.function
def overlay_y_in_X(X, y):
    '''
    Overlay one-hot-encoded `y` in one sample of X.
    
    Args:
        X: the shape is IMG_SHAPE
        y: int
        
    Returns:
        X: the shape is IMG_SHAPE
        y: int
    '''
    X = X * OVERLAY_BITMASK + tf.gather(OVERLAY_LABELS, y)
    return X, y

@tf.function
def randomize_y(X, y):
    '''
    Generate randomized `y` for one sample.
    '''
    y = tf.random.uniform(tf.shape(y), maxval=NUM_CLASS, dtype=tf.int32)
    
    return X, y

@tf.function
def overlay_default_in_X(X):
    '''
    Overlay the default label in a batch of X.
    
    Args:
        X: the shape is `(m, ) + IMG_SHAPE`
        
    Returns:
        X: the shape is `(m, ) + IMG_SHAPE`
    '''
    X = X * OVERLAY_BITMASK + OVERLAY_DEFAULT
    return X

@tf.function
def overlay_each_label_in_one_X(X):
    '''
    Replicate `X` `NUM_CLASS` times, and overlay on each copy one of the
    labels. 
    
    Args:
        X: the shape is `(NUM_IMGS, ) + IMG_SHAPE`
        
    Returns:
        _X: the shape is `(NUM_CLASS * NUM_IMGS, ) + IMG_SHAPE`
        
    '''
    _X = tf.repeat(X, NUM_CLASS, axis=0)
    _y = tf.tile(tf.range(NUM_CLASS), [tf.shape(X)[0], ])
    _X = overlay_y_in_X(_X, _y)[0]
    return _X

@tf.function
def create_eval_X(X, is_unsupervised):
    '''
    Based on the data `X` of size (m, ) + IMG_SHAPE, two new parts of 
    data is created for the purpose of evaluations.

    Part one depends on whether the model is accepting "supervised data"
    (`X` that carries labels) or "unsupervised data". If it is 
    "supervised", then a default label (e.g. `[0.2, 0.2, 0.2, 0.2, 0.2]`
    when `NUM_CLASS=5`) is overlayed in the first 5 elements of each 
    sample of `X`. If it is unsupervised, then the original `X` is used.

    Part two is to copy `X` `NUM_CLASS` times, and in each copy, overlay
    a one-hot-encoded label (e.g. `[1., 0., 0., 0., 0.]`).

    Part one takes the shape of (m, ) + IMG_SHAPE, and part two takes
    the shape of (m * NUM_CLASS, ) + IMG_SHAPE. The two parts are
    stacked together, and are ready to be passed to the model for
    evaluation.
    '''
    if is_unsupervised:
        part1 = X
    else:
        part1 = overlay_default_in_X(X)
        
    part2 = overlay_each_label_in_one_X(X)
    
    return tf.concat([part1, part2], 0)

@tf.function
def eval_unsupervised(X, y):
    return create_eval_X(X, True), y

@tf.function
def eval_supervised(X, y):
    return create_eval_X(X, False), y