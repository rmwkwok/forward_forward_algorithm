import tensorflow as tf

# Utils
def goodness(x, threshold=0.):
    return tf.reduce_sum(x**2, axis=-1) - tf.cast(threshold, tf.float32)

def preNorm(X):
    axis = tf.range(tf.rank(X))[1:]
    norm = tf.math.sqrt(tf.reduce_sum(X**2, axis=axis, keepdims=True))
    return X/(norm + 1e-7)

# Loss and Metric For FF-trained layers (e.g. FFDense).
# A softmax layer is not trained FF-wise, so it does not use them.
class FFLoss(tf.keras.losses.BinaryCrossentropy):
    def __init__(self, threshold, **kwargs):
        super().__init__(from_logits=True, **kwargs)
        self.threshold = threshold
    
    def __call__(self, y_true, y_pred):
        y_pred = goodness(y_pred, self.threshold)
        return super().__call__(y_true, y_pred)
        
class FFMetric(tf.keras.metrics.BinaryCrossentropy):
    def __init__(self, **kwargs):
        super().__init__(from_logits=True, **kwargs)
        
    def update_state(self, y_true, y_pred):
        y_pred = goodness(y_pred)
        return super().update_state(y_true, y_pred)
    
# FFLayers
# The base class comes first, followed by FFLayers inheriting the 
# base class and a tf.keras.layers.Layer.
class BaseFFLayer:
    TASK_TRANSFORM  = 'TASK_TRANSFORM'
    TASK_TRAIN_POS  = 'TASK_TRAIN_POS'
    TASK_TRAIN_NEG  = 'TASK_TRAIN_NEG'
    TASK_EVAL_POS   = 'TASK_EVAL_POS'
    TASK_EVAL_DUPED = 'TASK_EVAL_DUPED'
    
    def __init__(self, optimizer=None,  metric=None, metric_duped=None,
                 loss_pos=None, loss_neg=None, **kwargs):
        '''
        Turn a Tensorflow layer into a FFLayer by defining a class
        inheriting this class and the Tensorflow layer's class. 
        
        5 tasks are predefined. They accept `X` and `y_true` as input,
        and produces `y_pred` as output: 
            - transform: e.g. tf.keras.layers.Dense(X)
            - train_pos: optimize on positive pass data
            - train_neg: optimize on negative pass data
            - eval: evaluate layer
            - eval_duped: evaluate layer while expecting "duplicated"
                  data. At evaluation, data is duplicated before gets 
                  overlayed with different labels is passed through. 
                  This is for supervised-wise FF training.
                  
        The `ff_do_task()` calls the default task which is settable by
        calling the `ff_set_task()` method. This arrangement is for the
        sake of building different tensorflow graphs with even the same
        python function that calls the `ff_do_task()`
        
        Args:
            optimizer: `tf.keras.optimizers.Optimizer` object. Used in
                `TASK_TRAIN_POS` and `TASK_TRAIN_NEG`.
            metric: `tf.keras.metrics.Metric` object. Used in 
                `TASK_EVAL_POS`.
            metric_duped: `tf.keras.metrics.Metric` object. Used in 
                `TASK_EVAL_DUPED`.
            loss_pos: `tf.keras.losses.Loss` object. Used in 
                `TASK_TRAIN_POS`.
            loss_neg: `tf.keras.losses.Loss` object. Used in 
                `TASK_TRAIN_NEG`.
            **kwargs: passed to the inherited `tf.keras.layers.Layer` 
                object by the FFLayer that inherits this class.
        '''
        super().__init__(**kwargs)
        self.ff_opt = optimizer
        self.ff_metric = metric
        self.ff_metric_duped = metric_duped
        self.ff_loss_pos = loss_pos
        self.ff_loss_neg = loss_neg
        self.ff_task_fn = self.ff_task_transform
        self.gen_tasks_list()
        
    def gen_tasks_list(self):
        self.tasks = {
            self.TASK_TRANSFORM:  self.ff_task_transform, 
            self.TASK_TRAIN_POS:  self.ff_task_train_pos  if self.ff_loss_pos     is not None else self.ff_task_transform, 
            self.TASK_TRAIN_NEG:  self.ff_task_train_neg  if self.ff_loss_neg     is not None else self.ff_task_transform,
            self.TASK_EVAL_POS:   self.ff_task_eval_pos   if self.ff_metric       is not None else self.ff_task_transform, 
            self.TASK_EVAL_DUPED: self.ff_task_eval_duped if self.ff_metric_duped is not None else self.ff_task_transform, 
        }
        
    def ff_set_task(self, task):
        '''
        Set the default task to be calling `ff_do_task`.
        
        Args:
            task: one of `'TASK_TRANSFORM'`, `'TASK_TRAIN_POS'`,
                `'TASK_TRAIN_NEG'`, `'TASK_EVAL_POS'`, 
                `'TASK_EVAL_DUPED'` defined in this class. For example, 
                `BaseFFLayer.TASK_TRANSFORM`.
            
        Returns
            y_pred: `Tensor`. Transformation of `X`.
        '''
        self.ff_task_fn = self.tasks[task]
    
    def ff_do_task(self, X, y_true):
        '''
        Calls the default task as set by calling `ff_set_task`.
        
        Args:
            X: `Tensor`. Input data.
            y_true: `Tensor`. Target data.
            
        Returns
            y_pred: `Tensor`. Transformation of `X`.
        '''
        return self.ff_task_fn(X, y_true)
    
    def ff_task_train_pos(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self._ff_call_loss_pos(y_true, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred
    
    def ff_task_train_neg(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self._ff_call_loss_neg(y_true, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred
    
    def ff_task_eval_pos(self, X, y_true):
        y_pred = self(X)
        self.ff_metric.update_state(y_true, y_pred)
        return y_pred
    
    def ff_task_eval_duped(self, X, y_true):
        m = tf.shape(y_true)[0]
        y_pred = self(X)
        y_pred = tf.reshape(y_pred, (m, -1))
        self.ff_metric_duped.update_state(y_true, y_pred)
        return y_pred
    
    def ff_task_transform(self, X, y_true):
        return self(X)
    
    def _ff_call_loss_pos(self, y_true, y_pred):
        return self.ff_loss_pos(y_true, y_pred)
    
    def _ff_call_loss_neg(self, y_true, y_pred):
        return self.ff_loss_neg(y_true, y_pred)
        
    
# FFLayers inheriting the base class and a tf.keras.layers.Layer
class FFDense(BaseFFLayer, tf.keras.layers.Dense):
    def __init__(self, th_pos, th_neg, **kwargs):
        super().__init__(
            activation='relu', metric=FFMetric(), 
            loss_pos=FFLoss(th_pos), loss_neg=FFLoss(th_neg), **kwargs)
        
class FFSoftmax(BaseFFLayer, tf.keras.layers.Dense):
    def __init__(self, **kwargs):
        super().__init__(
            loss_pos=tf.keras.losses.SparseCategoricalCrossentropy(True),
            metric=tf.keras.metrics.SparseCategoricalAccuracy(), 
            **kwargs)
        
class FFGoodness(BaseFFLayer, tf.keras.layers.Lambda):
    def __init__(self, **kwargs):
        super().__init__(
            function=goodness,
            metric_duped=tf.keras.metrics.SparseCategoricalAccuracy(), 
            **kwargs)
        
class FFOverlay(BaseFFLayer, tf.keras.layers.Layer):
    def __init__(self, embedding, **kwargs):
        '''
        When `ff_task_eval_pos` or `ff_task_transform` is called, it 
        overlays an embedding onto a sample based on the sample's 
        `y_true`. When `ff_task_eval_duped` is called, it overlays 
        all embeddings onto each sample, transforming an `X` from 
        shape `(samples, features)` to 
        `(samples * embeddings, features)`.
        
        Args:
            embedding. a `Tensor` of shape 
                `(number of embeddings, features)`.
        '''
        super().__init__(metric_duped='dummy', **kwargs)
        self.ff_embedding = tf.cast(embedding, tf.float32)
        self.ff_emb_shape = tf.shape(self.embedding)[1:]
    
    def ff_task_transform(self, X, y_true):
        y_pred = X + tf.gather(self.embedding, y_true)
        return y_pred

    def ff_task_eval_pos(self, X, y_true):
        return self.ff_task_transform(X, y_true)
        
    def ff_task_eval_duped(self, X, y_true):
        y_pred = tf.expand_dims(X, 1) + self.embedding
        y_pred = tf.reshape(y_pred, (-1, *tf.unstack(self.emb_shape)))
        return y_pred
        
class FFPreNorm(BaseFFLayer, tf.keras.layers.Lambda):
    def __init__(self, **kwargs):
        super().__init__(function=preNorm, **kwargs)