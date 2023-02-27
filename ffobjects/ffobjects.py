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
    TASK_TRANSFORM      = 'TASK_TRANSFORM'
    TASK_TRAIN_POS      = 'TASK_TRAIN_POS'
    TASK_TRAIN_NEG      = 'TASK_TRAIN_NEG'
    TASK_EVAL_POS       = 'TASK_EVAL_POS'
    TASK_EVAL_NEG       = 'TASK_EVAL_NEG'
    TASK_EVAL_DUPED_POS = 'TASK_EVAL_DUPED_POS'

    def __init__(
        self,
        tfl,
        optimizer=None,
        metric=None,
        metric_duped=None,
        loss_pos=None,
        loss_neg=None,
        **kwargs
    ):
        '''
        6 tasks are predefined. They accept `X` and `y_true` as input,
        and produces `y_pred` as output:
            - transform: e.g. tf.keras.layers.Dense(X)
            - train_pos: optimize on positive pass data
            - train_neg: optimize on negative pass data
            - eval_pos: evaluate layer
            - eval_neg: evaluate layer
            - eval_duped_pos: evaluate layer while expecting 
                "duplicated" data. At evaluation, data is duplicated
                before gets overlayed with different labels is passed
                through. This is for supervised-wise FF training.

        The `ff_do_task()` calls the default task which is settable by
        calling the `ff_set_task()` method. This arrangement is for the
        sake of building different tensorflow graphs with even the same
        python function that calls the `ff_do_task()`

        Args:
            tfl: a `tf.keras.layers.Layer` object.
            optimizer: `tf.keras.optimizers.Optimizer` object. Used in
                `TASK_TRAIN_POS` and `TASK_TRAIN_NEG`.
            metric: `tf.keras.metrics.Metric` object. Used in
                `TASK_EVAL_POS`.
            metric_duped: `tf.keras.metrics.Metric` object. Used in
                `TASK_EVAL_DUPED_POS`.
            loss_pos: `tf.keras.losses.Loss` object. Used in
                `TASK_TRAIN_POS`.
            loss_neg: `tf.keras.losses.Loss` object. Used in
                `TASK_TRAIN_NEG`.
            **kwargs: passed to the inherited `tf.keras.layers.Layer`
                object by the FFLayer that inherits this class.
        '''
        self.tfl = tfl
        self.ff_opt = optimizer
        self.ff_metric = metric
        self.ff_metric_duped = metric_duped
        self.ff_loss_pos = loss_pos
        self.ff_loss_neg = loss_neg
        self.ff_task_fn = self.ff_task_transform
        self.tasks = {
            self.TASK_TRANSFORM:      self.ff_task_transform,
            self.TASK_TRAIN_POS:      self.ff_task_train_pos,
            self.TASK_TRAIN_NEG:      self.ff_task_train_neg,
            self.TASK_EVAL_POS:       self.ff_task_eval_pos,
            self.TASK_EVAL_NEG:       self.ff_task_eval_neg,
            self.TASK_EVAL_DUPED_POS: self.ff_task_eval_duped_pos,
        }

    def ff_set_task(self, task):
        '''
        Set the default task.
        '''
        self.ff_task_fn = self.tasks[task]
    
    def ff_do_task(self, X, y_true=None):
        '''
        Calls the default task.
        '''
        return self.ff_task_fn(X, y_true)

    def ff_task_transform(self, X, y_true=None):
        return self(X)

    def ff_task_train_pos(self, X, y_true=None):
        return self(X)

    def ff_task_train_neg(self, X, y_true=None):
        return self(X)

    def ff_task_eval_pos(self, X, y_true=None):
        return self(X)

    def ff_task_eval_neg(self, X, y_true=None):
        return self(X)

    def ff_task_eval_duped_pos(self, X, y_true=None):
        return self(X)

    def __call__(self, *args, **kwargs):
        return self.tfl(*args, **kwargs)
    
    @property
    def trainable_weights(self):
        return self.tfl.trainable_weights

# FFLayers inheriting the base class and a tf.keras.layers.Layer
class FFDense(BaseFFLayer):
    def __init__(self, th_pos, th_neg, optimizer, **tfl_kwargs):
        super().__init__(
            tfl=tf.keras.layers.Dense(activation='relu', **tfl_kwargs),
            optimizer=optimizer,
            metric=FFMetric(),
            loss_pos=FFLoss(th_pos), 
            loss_neg=FFLoss(th_neg), 
        )

    def ff_task_train_pos(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self.ff_loss_pos(y_true, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred

    def ff_task_train_neg(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self.ff_loss_neg(y_true, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred

    def ff_task_eval_pos(self, X, y_true):
        y_pred = self(X)
        self.ff_metric.update_state(y_true, y_pred)
        return y_pred

    def ff_task_eval_neg(self, X, y_true):
        y_pred = self(X)
        self.ff_metric.update_state(y_true, y_pred)
        return y_pred

class FFSoftmax(BaseFFLayer):
    def __init__(self, optimizer, **tfl_kwargs):
        super().__init__(
            tfl=tf.keras.layers.Dense(activation='linear', **tfl_kwargs),
            optimizer=optimizer,
            metric=tf.keras.metrics.SparseCategoricalAccuracy(),
            loss_pos=tf.keras.losses.SparseCategoricalCrossentropy(True),
        )

    def ff_task_train_pos(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            loss = self.ff_loss_pos(y_true, y_pred)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred

    def ff_task_eval_pos(self, X, y_true):
        y_pred = self(X)
        self.ff_metric.update_state(y_true, y_pred)
        return y_pred

class FFGoodness(BaseFFLayer):
    def __init__(self, **tfl_kwargs):
        super().__init__(
            tfl=tf.keras.layers.Lambda(goodness, **tfl_kwargs),
            metric_duped=tf.keras.metrics.SparseCategoricalAccuracy(),
        )

    def ff_task_eval_duped_pos(self, X, y_true):
        m = tf.shape(y_true)[0]
        y_pred = self(X)
        y_pred = tf.reshape(y_pred, (m, -1))
        self.ff_metric_duped.update_state(y_true, y_pred)
        return y_pred

class FFOverlay(BaseFFLayer):
    def __init__(self, embedding, **tfl_kwargs):
        '''
        When `ff_task_eval_pos` or `ff_task_transform` is called, it
        overlays an embedding onto a sample based on the sample's
        `y_true`. When `ff_task_eval_duped_pos` is called, it overlays
        all embeddings onto each sample, transforming an `X` from
        shape `(samples, features)` to
        `(samples * embeddings, features)`.

        Args:
            embedding. a `Tensor` of shape
                `(number of embeddings, features)`.
        '''
        self.ff_embedding = tf.cast(embedding, tf.float32)
        self.ff_emb_shape = tf.shape(self.ff_embedding)[1:]
        
        def function(X):
            X, y_true = X
            return X + tf.gather(self.ff_embedding, y_true)

        super().__init__(
            tfl=tf.keras.layers.Lambda(function, **tfl_kwargs),
        )

    def ff_task_eval_duped_pos(self, X, y_true=None):
        X, y_true = X
        y_pred = tf.expand_dims(X, 1) + self.ff_embedding
        y_pred = tf.reshape(y_pred, (-1, *tf.unstack(self.ff_emb_shape)))
        return y_pred

class FFPreNorm(BaseFFLayer):
    def __init__(self, **tfl_kwargs):
        super().__init__(
            tfl=tf.keras.layers.Lambda(preNorm, **tfl_kwargs),
        )

class FFRoutedDense(BaseFFLayer):
    def __init__(self, th_pos, th_neg, optimizer, **tfl_kwargs):
        super().__init__(
            tfl=tf.keras.layers.Dense(activation='relu', **tfl_kwargs),
            optimizer=optimizer,
            metric=FFMetric(),
            loss_pos=FFLoss(th_pos), 
            loss_neg=FFLoss(th_neg), 
        )

    def ff_set_ctu_map(self, ctu_map_pos, ctu_map_neg):
        self._ff_ctu_map_pos = ctu_map_pos
        self._ff_ctu_map_neg = ctu_map_neg
        return self

    def ff_set_classes(self, classes):
        self._ff_classes = classes
        return self

    def _ff_route_y_pred(self, y_pred, ctu_map):
        route = tf.nn.embedding_lookup(ctu_map, self._ff_classes)
        y_pred_routed = y_pred * route
        return y_pred_routed

    def ff_task_train_pos(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            y_pred_routed = self._ff_route_y_pred(y_pred, self._ff_ctu_map_pos)
            loss = self.ff_loss_pos(y_true, y_pred_routed)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred

    def ff_task_train_neg(self, X, y_true):
        with tf.GradientTape() as tape:
            y_pred = self(X)
            y_pred_routed = self._ff_route_y_pred(y_pred, self._ff_ctu_map_neg)
            loss = self.ff_loss_neg(y_true, y_pred_routed)
        grads = tape.gradient(loss, self.trainable_weights)
        self.ff_opt.apply_gradients(zip(grads, self.trainable_weights))
        return y_pred

    def ff_task_eval_pos(self, X, y_true):
        y_pred = self(X)
        y_pred_routed = self._ff_route_y_pred(y_pred, self._ff_ctu_map_pos)
        self.ff_metric.update_state(y_true, y_pred_routed)
        return y_pred

    def ff_task_eval_neg(self, X, y_true):
        y_pred = self(X)
        y_pred_routed = self._ff_route_y_pred(y_pred, self._ff_ctu_map_neg)
        self.ff_metric.update_state(y_true, y_pred_routed)
        return y_pred

# class FFClassFilter(BaseFFLayer):
#     def __init__(self, keep_classes, num_classes, **tfl_kwargs):
#         self.ff_keep_classes = tf.transpose(
#                                    tf.reduce_sum(
#                                        tf.one_hot(keep_classes, num_classes),
#                                        axis=0,
#                                        keepdims=True))

#         def function(X):
#             X, y_true = X
#             arg = tf.equal(
#                       tf.squeeze(
#                           tf.one_hot(y_true, num_classes) @\
#                           self.ff_keep_classes), 1.)
#             X = tf.boolean_mask(X, arg)
#             y_true = tf.boolean_mask(y_true, arg)
#             return (X, y_true)

#         super().__init__(
#             tfl=tf.keras.layers.Lambda(function, **tfl_kwargs),
#         )

# class FFGoodness2(BaseFFLayer):
#     def __init__(self, **tfl_kwargs):
#         super().__init__(
#             tfl=tf.keras.layers.Layer,
#             metric=tf.keras.metrics.SparseCategoricalAccuracy(),
#             **tfl_kwargs)

#     def ff_task_eval_pos(self, X, y_true):
#         m = tf.shape(y_true)[0]
#         y_pred = X
#         y_pred = tf.reshape(y_pred, (m, 10, -1))
#         y_pred = goodness(y_pred)
#         self.ff_metric.update_state(y_true, y_pred)
#         return y_pred