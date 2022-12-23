import tensorflow as tf
from utils import create_eval_X, NUM_CLASS, preNorm

'''
This work attempts to reproduce the forward-forward algorithm in this
paper:
Author: Geoffrey Hinton. 
Title: The Forward-Forward Algorithm: Some Preliminary Investigations
Link: https://www.cs.toronto.edu/~hinton/FFA13.pdf
'''

class FFLoss(tf.keras.losses.BinaryCrossentropy):
    '''
    Implementing the goodness function suggested by the paper.
    '''
    
    def __init__(self, threshold):
        super().__init__(from_logits=True, name='ff_binary_crossentropy')
        self.threshold = tf.cast(threshold, tf.float32)
        
    def __call__(self, y_true, y_pred):
        axis = tf.range(tf.rank(y_pred))[1:]
        y_pred = tf.reduce_sum(y_pred**2, axis=axis)
        y_pred = y_pred - self.threshold
        return super().__call__(y_true, y_pred)

class FFMetric(tf.keras.metrics.BinaryCrossentropy):
    '''
    Implementing the goodness function suggested by the paper.
    '''
    
    def __init__(self, threshold):
        super().__init__(from_logits=True, name='ff_binary_crossentropy')
        self.threshold = tf.cast(threshold, tf.float32)
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        axis = tf.range(tf.rank(y_pred))[1:]
        y_pred = tf.reduce_sum(y_pred**2, axis=axis)
        y_pred = y_pred - self.threshold
        return super().update_state(y_true, y_pred, sample_weight)
        
def FFLayer(layer_class, **kwargs):
    '''
    Creating a layer inheriting the `layer_class`, and carries the 
    layer's own metric, loss and optimizer objects. If the loss or the
    optimizer is not defined, the layer won't be trained. `do_ff` is
    needed to specify whether the layer performs a forward-forward 
    training or a single forward training. 

    Example:
        This is a FF Dense layer. `do_ff`, `optimizer`, and `loss_fn` 
        are required for it to be a trainable FF layer. 

        ```
        x = FFLayer(tf.keras.layers.Dense, units=32, activation='relu', 
            do_ff=True, optimizer=Adam(0.0001), 
            loss_fn=FFLoss(threshold=0.),
            )
        ```

        For a trainable non-FF layer, only the `optimizer` and `loss_fn`
        are required.

    Args:
        layer_class: A class inheriting `tf.keras.layers.Layer`
        **kwargs: arguments passed to the `layer_class` 

        do_ff: A `bool` indicating whether this layer performs 
            forward-forward training (True) or forward training (False).
            A softmax layer usually does forward training.

        optimizer: A `tf.keras.optimizers.Optimizer` object

        loss_fn: A `tf.keras.losses.Loss` object
        
        metric: A `tf.keras.metrics.Meric` object
        
        is_goodness_softmax: If `True`, at evaluation, this layer is
            expected to have output of shape `(None, )`. For example,
            the layer is a Lambda layer which computes the goodness of a
            Dense input with `tf.reduce_sum(X**2, axis=-1)`. Then the
            evaluation samples will be replicated `NUM_CLASS` times and
            each copy is overlayed with a different one-hot encoded 
            label. The output of the layer is reshaped to be
            `(None, NUM_CLASS)` to perform a softmax-like evalatuion 
            with the metric function defined in `metric`

    Returns:
        A `Layer` object inheriting the `layer_class` which can be used
        to build tensorflow model
    '''

    class Layer(layer_class):
        def __init__(self, do_ff=False, optimizer=None, loss_fn=None, 
                     metric=None, is_goodness_softmax=False, **kwargs):
            super().__init__(**kwargs)
            self.ff_do_ff = do_ff
            self.ff_metric = metric
            self.ff_loss_fn = loss_fn
            self.ff_optimizer = optimizer
            self.ff_is_goodness_softmax = is_goodness_softmax
            
        def ff_reset_metric(self):
            if self.ff_metric:
                self.ff_metric.reset_state()
                
        def ff_get_metric_results(self, results_dict):
            if self.ff_metric:
                results_dict[self.name] = self.ff_metric.result().numpy()
            
    Layer.__name__ = layer_class.__name__
    
    layer_object = Layer(**kwargs)
    
    if isinstance(layer_object, tf.keras.layers.InputLayer):
        # Copied from https://github.com/keras-team/keras/blob/e6784e4302c7b8cd116b74a784f4b78d60e83c26/keras/engine/input_layer.py#L442-L446
        outputs = layer_object._inbound_nodes[0].outputs
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        else:
            return outputs
    else:
        return layer_object


class FFModel(tf.keras.Model):
    '''
    Create a FFModel inheriting `tf.keras.Model`. It is expected that
    the model is created by linking the FFLayer Functional APIs 
    (inheriting Tensorflow), because we are expected to include all
    trainable and evaluated layers in the list of output of the model in
    order for the `FFModel.train` call to train them. 
    `tf.keras.Sequential` does not allow us to customize the list of
    output nodes.

    Example:

    This example creates a 3-layer NN where the hidden layers are FF 
    trainable, whereas the output layer is trainable but not FF. All 
    trainable layers, namely `y1, y2, y4`, and aevaluated layers, namely 
    `y3, y4` are all added to the output when calling `FFModel`.

    ```
    x0 = FFLayer(tf.keras.layers.InputLayer, input_shape=(100, ))

    y1 = FFLayer(tf.keras.layers.Dense, units=32, activation='relu', 
                 do_ff=True, optimizer=tf.keras.optimizers.Adam(0.0001), 
                 loss_fn=FFLoss(threshold=0.),
                 name='dense 1')(x0)

    y2 = FFLayer(tf.keras.layers.Dense, units=16, activation='relu', 
                 do_ff=True, optimizer=tf.keras.optimizers.Adam(0.0001), 
                 loss_fn=FFLoss(threshold=0.),
                 name='dense 2')(y1)
        
    y3 = FFLayer(tf.keras.layers.Concatenate,
                 metric=tf.keras.metrics.SparseCategoricalAccuracy(), 
                 is_goodness_softmax=True,
                 name='goodness softmax')([y1, y2])

    y4 = FFLayer(tf.keras.layers.Dense, units=10, activation='linear', 
                 do_ff=False, optimizer=tf.keras.optimizers.Adam(0.0001), 
                 loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                 metric=tf.keras.metrics.SparseCategoricalAccuracy(), 
                 name='dense softmax'
                 )(y2)

    model = FFModel(x0, [y1, y2, y3, y4])
    ```
    
    To train the model, call `.train(...)` instead of `.fit(...)`. 

    ```
    model.train(ds_train_pos, ds_train_neg, ds_valid_pos, ds_valid_neg, 
                epochs=10, eval_every=1, 
                report=['pos_pass/goodness softmax', 
                        'pos_pass/dense softmax']
    ```

    `ds_train_pos`, `ds_train_neg`, `ds_valid_pos`, `ds_valid_neg` are 
    iterables of `(X, y)`. `ds_train_pos` and `ds_train_neg` are for 
    training, and all four are used for evaluation. 
    
    Evaluation metrics are defined in each layer, and if 
    `is_goodness_softmax=False` is passed to the layer, the evaluation
    is based on the the data produced in the input data iterables, 
    otherwise, the evaluation is on modified dataset that has 
    one-hot-encoded labels overlayed.
    
    We can choose to report which metric results at printing an epoch's 
    progress, and it is in a list of strings where each string should 
    have the format of 'pos_pass/<layer_name>' or 
    'neg_pass/<layer_name>' to indicate that it is a result of which
    layer and is on a positive or a negative dataset. For every item in 
    `report`, it prints the training set result followed by the
    validation set's.
    '''
    def __init__(self, is_unsupervised=True, *args, **kwargs):
        '''
        Building a `FFModel` inheriting `tf.keras.Model`.

        Args:
            is_unsupervised: whether it will be accepting unsupervised 
                data for the FF layers in the model. The affects the 
                evaluation step. 
            *args, **kwargs: Passed into `tf.keras.Model`.

        Returns:
            A `FFModel`.
        '''
        super().__init__(*args, **kwargs)
        self.ff_is_unsupervised = is_unsupervised
        self.ff_layers = [self.get_layer(n) for n in self.output_names]
    
    def ff_reset_all_metrics(self):
        for layer in self.ff_layers:
            layer.ff_reset_metric()
    
    def ff_get_all_metric_results(self, results_dict):
        for layer in self.ff_layers:
            layer.ff_get_metric_results(results_dict)
            
    def ff_print_record(self, record, report):
        '''
        At the end of an epoch, print metric results chosen in `report`.
        '''
        epoch = record['epoch']
        
        string = ''
        for item in report:
            _pass, layer_name = item.split('/')
            temp = []
            for tv in ['train', 'valid']:
                for pn in ['pos_pass', 'neg_pass']:
                    if layer_name in record[tv][pn] and pn == _pass:
                        x = record[tv][pn][layer_name]
                        temp.append(f'{x:.6f}')
                        
            if len(temp):
                temp = ' '.join(temp)
                string = f'{string} | {layer_name} {temp}'
        
        print(f'epoch {epoch: 5d}{string}')
            
    def train(self, ds_train_pos, ds_train_neg, ds_valid_pos, ds_valid_neg, 
              epochs, eval_every=5, report=[]):
        '''
        train the model. Use this method instead of `.fit(...)` for
        forward-forward algorithm. Use `.fit(...)` for backprop 
        algorithm.
        
        Args:
            ds_train_pos, ds_train_neg, ds_valid_pos, ds_valid_neg: 
                iterables of (X, y)
            epochs: int. Number of training epochs
            eval_every: int. Evaluate once every N epochs.
            report: list of strings. A list of metric to report at the 
                end of an epoch. The string should the format of 
                'pos_pass/<layer_name>' or 'neg_pass/<layer_name>' to
                indicate that it is a result of which layer and is on a
                positive or a negative dataset.

        Returns:
            history: a history of all evaluation results, reported or 
                not.
        '''
        history = []
        
        do_evaluate = lambda e: e % eval_every == 0 or e == epochs-1
        
        ds_train = [('pos_pass', ds_train_pos), ('neg_pass', ds_train_neg)]
        ds_valid = [('pos_pass', ds_valid_pos), ('neg_pass', ds_valid_neg)]
        datasets = [('train', ds_train), ('valid', ds_valid)]
        
        for epoch in range(epochs):
            
            # Gradient descent
            for _pass, dataset in ds_train:
                for X, y_true in dataset:
                    self.ff_gradient_descent(X, y_true, _pass)
            
            # Evaluation
            if not do_evaluate(epoch):
                continue
                
            record = {'epoch': epoch,
                      'train': {'pos_pass': {}, 'neg_pass': {}, },
                      'valid': {'pos_pass': {}, 'neg_pass': {}, },}
            
            for tv, ds in datasets:
                for _pass, dataset in ds:
                    self.ff_reset_all_metrics()
                    for X, y_true in dataset:
                        self.ff_evaluate(X, y_true, _pass)
                    self.ff_get_all_metric_results(record[tv][_pass])
                    
            history.append(record)
            self.ff_print_record(record, report)

        return history
    
    @tf.function
    def ff_convert_label(self, y_true, layer, _pass):
        '''
        In a forward-forward algorithm layer, the label is always 1 in a
        positive pass, and 0 in a negative pass.
        '''
        if layer.ff_do_ff:
            if _pass == 'pos_pass':
                return y_true * 0 + 1 
            elif _pass == 'neg_pass':
                return y_true * 0 
        else:
            return y_true
        
    @tf.function
    def ff_gradient_descent(self, X, y_true, _pass):
        '''
        Training is skipped if it is not a forward-forward layer and it 
        is in a negative pass.
        '''
        with tf.GradientTape(persistent=True) as tape:
            losses = []
            for layer, y_pred in zip(self.ff_layers, self(X, training=True)):
                if layer.ff_loss_fn and\
                    (layer.ff_do_ff or _pass == 'pos_pass'):
                    yt = self.ff_convert_label(y_true, layer, _pass)
                    losses.append((layer, layer.ff_loss_fn(yt, y_pred)))
                
        for layer, loss in losses:
            if layer.ff_optimizer:
                grads = tape.gradient(loss, layer.trainable_weights)
                layer.ff_optimizer.apply_gradients(
                    zip(grads, layer.trainable_weights))

        del tape
    
    @tf.function
    def ff_evaluate(self, X, y_true, _pass):
        '''
        Based on the data `X` of size (m, ) + IMG_SHAPE, two new parts 
        of data is created. 
        
        Part one depends on whether the model is accepting 
        "supervised data" (`X` that carries labels) or "unsupervised 
        data". If it is "supervised", then a default label (e.g. 
        `[0.2, 0.2, 0.2, 0.2, 0.2]` when `NUM_CLASS=5`) is overlayed in 
        the first 5 elements of each sample of `X`. If it is 
        unsupervised, then the original `X` is used.
        
        Part two is to copy `X` `NUM_CLASS` times, and in each copy, 
        overlay a one-hot-encoded label (e.g. `[1., 0., 0., 0., 0.]`).
        
        Part one takes the shape of (m, ) + IMG_SHAPE, and part two 
        takes the shape of (m * NUM_CLASS, ) + IMG_SHAPE. The two parts 
        are stacked together, and passed to the model.
        
        At evaluation of a layer, if `is_goodmax_softmax=True` was 
        passed into the layer, it takes the model's output on part two, 
        reshape it back to `(m, NUM_CLASS)` and perform a softmax-like 
        evaluation; otherwise, the model's output on part one is used 
        for evaluation.
        '''
        m = tf.shape(X)[0]
        X = create_eval_X(X, self.ff_is_unsupervised)

        for name, y_pred in zip(self.output_names, self(X)):
            layer = self.get_layer(name)
            if layer.ff_do_ff or _pass == 'pos_pass':
                if layer.ff_metric:
                    if layer.ff_is_goodness_softmax:
                        yt = y_true
                        yp = tf.reshape(y_pred[m:], (m, NUM_CLASS))
                    else:
                        yt = self.ff_convert_label(y_true, layer, _pass)
                        yp = y_pred[:m]
                    
                    layer.ff_metric.update_state(yt, yp)
                
                    
                    
###############
### Layer blocks
###############

def ff_dense_block(units, activation, lr, th, idx, x):
    '''
          output1         output 2
            ^                ^
            |                |
    x --> Dense --> pre-Normalization
    ^
    |
    input
    
    Args:
        units, activation: passed to `tf.keras.layers.Dense`
        lr: learning rate
        th: FFLoss threshold
        idx: numbering for the block to be created
        x: input keras tensor to the block
        
    Returns:
        activity keras tensor and preNorm keras tensor
    '''
    
    activity = FFLayer(
        tf.keras.layers.Dense, units=units, activation=activation, 
        do_ff=True, 
        metric=FFMetric(threshold=th),
        loss_fn=FFLoss(threshold=th),
        optimizer=tf.keras.optimizers.Adam(lr), 
        name=f'b{idx}_dense',
    )(x)
    
    normalized = FFLayer(
        tf.keras.layers.Lambda, 
        function=preNorm,  
        name=f'b{idx}_prenorm',
    )(activity)
    
    return activity, normalized

def ff_conv2d_block(filters, kernel_size, activation, max_pool_size,
                    lr, th, idx, x):
    '''
    
                 output 1         output 2           output 3
                    ^                ^                  ^
                    |                |                  |
            ---> Flatten     ---> Flatten           ---------
            |                |                      |       |
    x --> Conv2D --> pre-Normalization -----> MaxPooling2D  |
    ^                        |                              |
    |                        --------->-no pooling-----------
    input
    
    Args:
        filters, kernel_size, activation: passed to 
            `tf.keras.layers.Conv2D`
        max_pool_size: passed to `tf.keras.layers.MaxPooling2D`. if 
            `None`, Max Pooling is not used
        lr: learning rate
        th: FFLoss threshold
        idx: numbering for the block to be created
        x: input keras tensor to the block
        
    Returns:
        activity keras tensor and preNorm keras tensor
    '''
    
    activity = FFLayer(
        tf.keras.layers.Conv2D, 
        filters=filters, kernel_size=kernel_size, activation=activation, 
        do_ff=True, 
        metric=FFMetric(threshold=th),
        loss_fn=FFLoss(threshold=th),
        optimizer=tf.keras.optimizers.Adam(lr), 
        name=f'b{idx}_conv2d',
    )(x)
    
    flattened_activity = FFLayer(
        tf.keras.layers.Flatten,
        name=f'b{idx}_flattened_activity'
    )(activity)
    
    normalized = FFLayer(
        tf.keras.layers.Lambda, 
        function=preNorm,  
        name=f'b{idx}_prenorm',
    )(activity)
    
    flattened_normalized = FFLayer(
        tf.keras.layers.Flatten,
        name=f'b{idx}_flattened_preNormed'
    )(normalized)
    
    pooled = normalized
    if max_pool_size:
        pooled = FFLayer(
            tf.keras.layers.MaxPooling2D,
            pool_size=max_pool_size,
            name=f'b{idx}_maxpool'
        )(normalized)
    
    return activity, flattened_activity, flattened_normalized, pooled