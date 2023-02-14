import tensorflow as tf

'''
This work attempts to reproduce the forward-forward algorithm in this
paper:
Author: Geoffrey Hinton. 
Title: The Forward-Forward Algorithm: Some Preliminary Investigations
Link: https://www.cs.toronto.edu/~hinton/FFA13.pdf
'''

class FFConstants:
    POS = 'pos'
    NEG = 'neg'
        
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
        
        is_goodness_softmax: Set it to `True` if the layer serves to 
            predict a class based on goodness, `False` otherwise.
            
        report_metric_pos: A `bool` indicating whether metric evaluation
            result for the positive pass should be reported during
            training
            
        report_metric_neg: A `bool` indicating whether metric evaluation
            result for the negative pass should be reported during
            training

    Returns:
        A `Layer` object inheriting the `layer_class` which can be used
        to build tensorflow model
    '''

    class Layer(layer_class):
        def __init__(self, do_ff=False, optimizer=None, loss_fn=None, 
                     metric=None, is_goodness_softmax=False, 
                     report_metric_pos=False, report_metric_neg=False,
                     **kwargs):
            super().__init__(**kwargs)
            self.ff_do_ff = do_ff
            self.ff_metric = metric
            self.ff_loss_fn = loss_fn
            self.ff_optimizer = optimizer
            self.ff_is_goodness_softmax = is_goodness_softmax
            self.ff_report_metric = {
                FFConstants.POS: report_metric_pos,
                FFConstants.NEG: report_metric_neg,
            }
            
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
    Create a FFModel inheriting `tf.keras.Model`. 

    Example:

    This example creates a 3-layer NN where the hidden layers are FF 
    trainable, whereas the output layer is trainable but not FF. All 
    trainable layers, namely `y1, y2, y4`, and all evaluated layers, 
    namely `y3, y4` are all added to the output when calling `FFModel`.

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
                 do_ff=False, 
                 optimizer=tf.keras.optimizers.Adam(0.0001), 
                 loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(
                             from_logits=True),
                 metric=tf.keras.metrics.SparseCategoricalAccuracy(), 
                 name='dense softmax'
                 )(y2)

    model = FFModel(x0, [y1, y2, y3, y4])
    ```
    
    To train the model, call `.ff_train(...)` instead of `.fit(...)`. 

    ```
    model.ff_train(ds_train, ds_valid_for_eval, epochs=10, eval_every=5)
    ```
    '''
    def __init__(self, *args, **kwargs):
        '''
        Building a `FFModel` inheriting `tf.keras.Model`.

        Args:
            *args, **kwargs: Passed into `tf.keras.Model`.

        Returns:
            A `FFModel`.
        '''
        super().__init__(*args, **kwargs)
        self.ff_layers = [self.get_layer(n) for n in self.output_names]
    
    def ff_reset_all_metrics(self):
        for layer in self.ff_layers:
            layer.ff_reset_metric()
    
    def ff_get_all_metric_results(self, results_dict):
        for layer in self.ff_layers:
            layer.ff_get_metric_results(results_dict)
            
    def ff_print_record(self, record):
        '''
        At the end of an epoch, print metric results.
        '''
        epoch = record['epoch']
        
        string = ''
        for layer in self.ff_layers:
            for pn in [FFConstants.POS, FFConstants.NEG]:
                
                temp = []
                for tv in ['train', 'valid']:
                    if layer.name in record[tv][pn] and\
                        layer.ff_report_metric[pn]:
                        x = record[tv][pn][layer.name]
                        temp.append(f'{x:.6f}')
                        
                if len(temp):
                    temp = ' '.join(temp)
                    string = f'{string} | {layer.name}/{pn} {temp}'
        
        print(f'epoch {epoch: 5d}{string}')
            
    def ff_train(self, ds_train, ds_valid_for_eval, epochs, eval_every=5):
        '''
        train the model. Use this method instead of `.fit(...)` for
        forward-forward algorithm. Use `.fit(...)` for backprop 
        algorithm.
        
        Args:
            ds_train: a tuple of of two training datasets. The first is
                for the positive pass, and the second the negative pass
            ds_valid_for_eval: a list of tuples. Each tuple has two
                evaluation datasets. The first is for the positive pass,
                and the second the negative pass
            epochs: int. Number of training epochs
            eval_every: int. Evaluate once every N epochs. Which layer's
                evaluation will be printed is controlled by the
                the layer's ff_report_metric_pos and 
                ff_report_metric_neg parameters

        Returns:
            history: a history of all evaluation results, reported or 
                not.
        '''
        history = []
        _passes = [FFConstants.POS, FFConstants.NEG]
        
        do_evaluate = lambda e: e % eval_every == 0 or e == epochs-1
        
        for epoch in range(epochs):
            
            # Gradient descent
            for _pass, dataset in zip(_passes, ds_train):
                for X, y_true in dataset:
                    self._ff_gradient_descent(X, y_true, _pass)
            
            # Evaluation
            if not do_evaluate(epoch):
                continue
                
            record = {'epoch': epoch,
                      'train': {FFConstants.POS: {}, FFConstants.NEG: {}, },
                      'valid': {FFConstants.POS: {}, FFConstants.NEG: {}, },}
            
            for tv, ds in ds_valid_for_eval:
                for _pass, dataset in zip(_passes, ds):
                    self.ff_reset_all_metrics()
                    for X, y_true in dataset:
                        self._ff_evaluate(X, y_true, _pass)
                    self.ff_get_all_metric_results(record[tv][_pass])
                    
            history.append(record)
            self.ff_print_record(record)

        return history
    
    @tf.function
    def _ff_convert_label(self, y_true, layer, _pass):
        '''
        In a forward-forward algorithm layer, the label is always 1 in a
        positive pass, and 0 in a negative pass.
        '''
        if layer.ff_do_ff:
            if _pass == FFConstants.POS:
                return y_true * 0 + 1 # always 1
            elif _pass == FFConstants.NEG:
                return y_true * 0 # always 0
        else:
            return y_true
        
    @tf.function
    def _ff_gradient_descent(self, X, y_true, _pass):
        with tf.GradientTape(persistent=True) as tape:
            losses = []
            for layer, y_pred in zip(self.ff_layers, self(X, training=True)):
                if layer.ff_loss_fn and\
                    (layer.ff_do_ff or _pass == FFConstants.POS):
                    yt = self._ff_convert_label(y_true, layer, _pass)
                    losses.append((layer, layer.ff_loss_fn(yt, y_pred)))
                
        for layer, loss in losses:
            if layer.ff_optimizer:
                grads = tape.gradient(loss, layer.trainable_weights)
                layer.ff_optimizer.apply_gradients(
                    zip(grads, layer.trainable_weights))

        del tape
    
    @tf.function
    def _ff_evaluate(self, X, y_true, _pass):
        for name, y_pred in zip(self.output_names, self(X)):
            layer = self.get_layer(name)
            if layer.ff_do_ff or _pass == FFConstants.POS:
                if layer.ff_metric:
                    if layer.ff_is_goodness_softmax:
                        yt = y_true
                    else:
                        yt = self._ff_convert_label(y_true, layer, _pass)
                    
                    layer.ff_metric.update_state(yt, y_pred)
