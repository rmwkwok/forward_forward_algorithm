import tensorflow as tf
from tools import overlay_each_label_in_one_X, overlay_y_in_X

###############
### Loss Objects
###############

class FFLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, threshold):
        super().__init__(from_logits=True)
        self.threshold = tf.cast(threshold, tf.float32)
        
    def __call__(self, y_true, y_pred):
        n_axes = tf.size(tf.shape(y_pred))
        y_pred = tf.reduce_sum(y_pred**2, axis=tf.range(n_axes)[1:]) - self.threshold # By paper
        return super().__call__(y_true, y_pred)

###############
### Layer Objects
###############

class FFLayer:
    '''
        Base layer that wraps any tf.keras.layers.Layer.
        Each trainable layer has its own loss function and optimizer.
        Layer like Flatten isn't trained, and is wrapped without loss and optimizer.
    '''
    def __init__(self, layer, loss=None, optimizer=None):
        self.layer = layer
        self.loss = loss
        self.optimizer = optimizer
        self.trainable = loss is not None and optimizer is not None
        
    @tf.function
    def train_step(self, X, y):
        with tf.GradientTape() as tape:
            layer_output = self.layer(X, training=True) #
            if not self.trainable:
                return layer_output
            
            loss_value = self.loss(y, layer_output)
        grads = tape.gradient(loss_value, self.layer.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.layer.trainable_weights))
        return layer_output

    @tf.function
    def train(self, X_pos, X_neg, y_pos, y_neg):
        '''
            Forward-forward training.
            Labels are always 1 for positive samples, and 0 for negative samples.
        '''
        X_pos = self.train_step(X_pos, y_pos*0+1)
        X_neg = self.train_step(X_neg, y_neg*0  )
        return X_pos, X_neg, y_pos, y_neg
    
# class FFDense(FFLayer):
#     def __init__(self, units, threshold, optimizer):
#         super().__init__(
#             layer=tf.keras.layers.Dense(units, activation='relu'), 
#             loss=FFLoss(threshold), 
#             optimizer=optimizer,
#         )
    
# class FFConv2D(FFLayer):
#     def __init__(self, filters, kernel_size, threshold, optimizer):
#         super().__init__(
#             layer=tf.keras.layers.Conv2D(filters, kernel_size, activation='relu'), 
#             loss=FFLoss(threshold), 
#             optimizer=optimizer,
#        )
    
class FFSoftmax(FFLayer):
    def __init__(self, units, optimizer):
        super().__init__(
            layer=tf.keras.layers.Dense(units, activation='linear'), 
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
            optimizer=optimizer
        )
    
    @tf.function
    def train(self, X_pos, X_neg, y_pos, y_neg):
        '''
            Softmax doesn't do FF training.
        '''
        X_pos = self.train_step(X_pos, y_pos)
        return X_pos, X_neg, y_pos, y_neg

###############
### Model Objects
###############

class FFModel:
    def __init__(self, layers):
        self.layers = layers
        
    @tf.function
    def predict(self, X):
        for layer in self.layers:
            X = layer.layer(X)
        return X
                    
    @tf.function
    def evaluate(self, dataset, metric):
        metric.reset_states()
        for X, _, y_true, _ in dataset:
            y_pred = self.predict(X)
            metric.update_state(y_true, y_pred)
        return metric.result()
            
    def train(self, ds_train, ds_valid, epochs, metric):
        for epoch in range(epochs):
            for batch in ds_train:
                for layer in self.layers:
                    batch = layer.train(*batch)
            if epoch % 5 == 0 or epoch == epochs - 1:
                eval_train = self.evaluate(ds_train, metric)
                eval_valid = self.evaluate(ds_valid, metric)
                print('epoch', epoch, 'train metric', eval_train.numpy(), 'valid metric', eval_valid.numpy())

class FFModel_Unsupervised(FFModel):
    def __init__(self, layers):
        super().__init__(layers)

class FFModel_Supervised(FFModel):
    def __init__(self, layers):
        super().__init__(layers)
          
    def train(self, ds_train, ds_valid, epochs, metric):
        ds_train = ds_train.map(overlay_y_in_X).cache()
        super().train(ds_train, ds_valid, epochs, metric)
        
    @tf.function
    def predict(self, X):
        goodness_accum = tf.zeros((tf.shape(X)[0], 10))
        X = overlay_each_label_in_one_X(X)
        for layer in self.layers:
            X = layer.layer(X)
            if isinstance(layer, FFLayer):
                axes = tf.range(tf.size(tf.shape(X)))[1:]
                goodness_accum += tf.transpose(tf.reshape(tf.reduce_sum(X, axis=axes), (10, -1)))
            
        return goodness_accum
