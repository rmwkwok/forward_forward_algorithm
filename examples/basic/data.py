import sys
sys.path.append('../../ffobjects/')

import numpy as np
import tensorflow as tf
from ffobjects import BaseFFLayer

def preprocess(Xy):
    '''
    Preprocessing. This example takes digits smaller than 5 as positive
    data, and the rest as negative data.
    Args:
        Xy: a tuple of `numpy.array` of data and targets.
        
    Returns:
        A positive `tuple` and a negative `tuple`. Each `tuple` has
        two `numpy.array` for data and targets respectively.
    '''
    X, y = Xy
    X = X.astype(np.float32) / 255.
    y = y.astype(np.int32)
    return (X[y< 5], y[y< 5]), (X[y>=5], y[y>=5])

def _create_dataset(X, y, y_ff, seed, batch_size):
    '''
    Args:
        X, y: `numpy.array` for data and targets.
        y_ff: `numpy.array`. It is `np.ones_like(...)` for positive data
            and `np.zeros_like(...)` for negative data.
        seed: When `None`, no shuffling is done. When `int`, reshuffling
            is done.
        batch_size: an `int` for the batch size.
    
    Returns:
        a `tf.data.Dataset`.
    '''
    ds = tf.data.Dataset.from_tensor_slices((X, y, y_ff))
    if seed is not None:
        ds = ds.shuffle(4096, seed, True)
    return ds.batch(batch_size, True, tf.data.AUTOTUNE, True)\
             .prefetch(tf.data.AUTOTUNE)

def create_mnist_datasets(seed=10, batch_size=128, include_duped=False):
    '''
    Args:
        batch_size: an `int` for batch size.
        include_duped: `bool`. Set to `True` for supervised-wise FF
            training; `False` for unsupervised-wise FF training.
        
    Returns:
        When `include_duped` is `False`, returns a list of three 
            datasets for the tasks of `TASK_TRAIN_POS`, `TASK_TRAIN_NEG`
            and `TASK_EVAL` respectively. When it is `True`, returns a
            list of four datasets by adding an extra one for 
            `TASK_EVAL_DUPED` which is needed to evaluate 
            supervised-wise FF layers.
    '''

    ((train_pos_X, train_pos_y), (train_neg_X, train_neg_y)), \
    ((valid_pos_X, valid_pos_y), (valid_neg_X, valid_neg_y)) =\
        map(preprocess, tf.keras.datasets.mnist.load_data())
    
    rng = np.random.default_rng(seed=seed)
    train_neg_y = rng.integers(0, 5, size=len(train_neg_y))

    train_ff_pos = _create_dataset(
        train_pos_X, train_pos_y, np.ones_like(train_pos_y), seed, batch_size)
    train_ff_neg = _create_dataset(
        train_neg_X, train_neg_y, np.zeros_like(train_neg_y), seed, batch_size)
    eval_ff_train = _create_dataset(
        train_pos_X, train_pos_y, np.ones_like(train_pos_y), seed, batch_size)
    eval_ff_valid = _create_dataset(
        valid_pos_X, valid_pos_y, np.ones_like(valid_pos_y), seed, batch_size)

    datasets = [
        # task, dataset name, dataset
        (BaseFFLayer.TASK_TRAIN_POS,  train_ff_pos), 
        (BaseFFLayer.TASK_TRAIN_NEG, train_ff_neg),
        # (BaseFFLayer.TASK_EVAL,  eval_ff_train),
        (BaseFFLayer.TASK_EVAL,  eval_ff_valid),
    ]
    
    if include_duped:
        datasets.extend([
            # (BaseFFLayer.TASK_EVAL_DUPED,  eval_ff_train),
            (BaseFFLayer.TASK_EVAL_DUPED,  eval_ff_valid),
        ])
    
    return datasets