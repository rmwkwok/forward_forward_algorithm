import tensorflow as tf
from tools import set_seed, preprocess
from ffobjects import FFLoss, FFModel_Unsupervised, FFModel_Supervised, FFLayer, FFSoftmax#, FFConv2D, FFDense

###############
### Data
###############

# configurations
batch_size = 128

# Load data - no test set
ds_train, ds_valid = map(tf.data.Dataset.from_tensor_slices, tf.keras.datasets.mnist.load_data())

# For FF algorithm, training set needs both pos and neg, so, 
# split data into pos and neg, and then combine them
ds_train_pos = ds_train.map(preprocess).take(30000).batch(batch_size, drop_remainder=True)
ds_train_neg = ds_train.map(preprocess).skip(30000).batch(batch_size, drop_remainder=True)
ds_valid_pos = ds_valid.map(preprocess).take(5000).batch(batch_size, drop_remainder=True)
ds_valid_neg = ds_valid.map(preprocess).skip(5000).batch(batch_size, drop_remainder=True)

ds_train = tf.data.Dataset\
    .choose_from_datasets([ds_train_pos, ds_train_neg], tf.data.Dataset.range(2).repeat())\
    .batch(2, drop_remainder=True)\
    .map(lambda X, y: (*tf.unstack(X), *tf.unstack(y)))\
    .cache()\
    .prefetch(10)

ds_valid = tf.data.Dataset\
    .choose_from_datasets([ds_valid_pos, ds_valid_neg], tf.data.Dataset.range(2).repeat())\
    .batch(2, drop_remainder=True)\
    .map(lambda X, y: (*tf.unstack(X), *tf.unstack(y)))\
    .cache()\
    .prefetch(10)

###############
### FF algorithm with Dense hidden layers
###############

set_seed()
preNormalization = lambda X: X/(tf.norm(X, keepdims=True, axis=-1) + 1e-7)

dense_supervised_ff = FFModel_Supervised([
    FFLayer(tf.keras.layers.Flatten()),
    FFLayer(tf.keras.layers.Dense(16, activation='relu'), FFLoss(0.), tf.keras.optimizers.Adam(0.0001)),
    FFLayer(tf.keras.layers.Lambda(preNormalization)),
    FFLayer(tf.keras.layers.Dense(10, activation='relu'), FFLoss(0.), tf.keras.optimizers.Adam(0.0001)),
])
dense_supervised_ff.train(ds_train, ds_valid, epochs=200, metric=tf.keras.metrics.SparseCategoricalAccuracy())

###############
### FF algorithm with Conv2D hidden layers
###############

set_seed()
preNormalization = lambda X: X/(tf.norm(X, keepdims=True, axis=-1) + 1e-7)

conv2d_supervised_ff = FFModel_Supervised([
    FFLayer(tf.keras.layers.Lambda(lambda X: tf.expand_dims(X, -1))),
    FFLayer(tf.keras.layers.Conv2D(4, (8, 8), activation='relu'), FFLoss(0.), tf.keras.optimizers.Adam(0.01)),
    FFLayer(tf.keras.layers.Lambda(preNormalization)),
    FFLayer(tf.keras.layers.Conv2D(8, (6, 6), activation='relu'), FFLoss(0.), tf.keras.optimizers.Adam(0.01)),
])
conv2d_supervised_ff.train(ds_train, ds_valid, epochs=5, metric=tf.keras.metrics.SparseCategoricalAccuracy())

