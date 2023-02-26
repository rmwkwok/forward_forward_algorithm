import numpy as np
import tensorflow as tf
from ffobjects import BaseFFLayer

NUM_CLASS = 10

def preprocess(Xy):
    X, y = Xy
    X = X.astype(np.float32) / 255.
    y = y.astype(np.int32)
    return X, y

def _create_dataset(X, y, y_ff, seed, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((X, y, y_ff))
    if seed is not None:
        ds = ds.shuffle(4096, seed, True)
    return ds.batch(batch_size, True, tf.data.AUTOTUNE, True)\
             .prefetch(tf.data.AUTOTUNE)

def create_mnist_datasets(seed=10, batch_size=128, is_supervised_ff=False):
    (train_X, train_y), (valid_X, valid_y) = \
        map(preprocess, tf.keras.datasets.mnist.load_data())
    
    train_y_neg = train_y if not is_supervised_ff else gen_fake_y(train_y)
    valid_y_neg = valid_y if not is_supervised_ff else gen_fake_y(valid_y)
    
    train_ff_pos = _create_dataset(
        train_X, train_y, 
        np.ones_like(train_y), seed, batch_size)
    train_ff_neg = _create_dataset(
        train_X, train_y_neg, 
        np.zeros_like(train_y), seed, batch_size)
    valid_ff_pos = _create_dataset(
        valid_X, valid_y, 
        np.ones_like(valid_y), seed, batch_size)
    valid_ff_neg = _create_dataset(
        valid_X, valid_y_neg, 
        np.zeros_like(valid_y), seed, batch_size)
    
    datasets = [
        (BaseFFLayer.TASK_TRAIN_POS,  train_ff_pos), 
        (BaseFFLayer.TASK_TRAIN_NEG, train_ff_neg),
        (BaseFFLayer.TASK_EVAL_POS,  valid_ff_pos),
    ]
    
    if is_supervised_ff:
        datasets.extend([
            (BaseFFLayer.TASK_EVAL_DUPED_POS,  valid_ff_pos),
        ])
    
    return datasets

def gen_fake_y(y_true, num_class=NUM_CLASS):
    n = len(y_true)
    y_zero = np.zeros(n, dtype=np.int32)
    all_classes = np.expand_dims(np.arange(num_class), 0)
    
    a = np.expand_dims(y_true, 1) != all_classes
    b = np.expand_dims(y_zero, 1) +  all_classes
    c = b[a].reshape((-1, num_class-1))
    d = np.random.randint(0, num_class-1, size=n)
    y_fake = c[np.arange(n), d]
    
    return y_fake