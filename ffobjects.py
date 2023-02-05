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
                
        @property
        def ff_is_pos_pass_trainable(self):
            return self.ff_loss_fn and self.ff_optimizer
                
        @property
        def ff_is_neg_pass_trainable(self):
            return self.ff_loss_fn and self.ff_optimizer and self.ff_do_ff
            
            
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ff_layers = [self.get_layer(n) for n in self.output_names]
        self.ff_pos_pass_trainable_layers = [l for l in self.ff_layers 
                                             if l.ff_is_pos_pass_trainable]
        self.ff_neg_pass_trainable_layers = [l for l in self.ff_layers 
                                             if l.ff_is_neg_pass_trainable]
            
    def ff_train(self, ds_train, ds_valid_for_eval, epochs, eval_every=5):
        history = []
        _passes = [FFConstants.POS, FFConstants.NEG]
        
        do_evaluate = lambda e: e % eval_every == 0 or e == epochs-1
        
        for epoch in range(epochs):
            
            # Positive pass Gradient descent
            for X, y_true in ds_train[0]:
                gradients = self._ff_pos_pass_compute_gradients(X, y_true)
                self._ff_pos_pass_update_params(gradients)
                
            # Negative pass Gradient descent
            for X, y_true in ds_train[1]:
                gradients = self._ff_neg_pass_compute_gradients(X, y_true)
                self._ff_neg_pass_update_params(gradients)
            
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
    
    ### Gradient Descent methods
        
    @tf.function
    def _ff_pos_pass_compute_gradients(self, X, y_true):
        losses = []
        gradients = []
        
        with tf.GradientTape(persistent=True) as tape:
            for layer, y_pred in zip(self.ff_layers, self(X, training=True)):
                if layer.ff_is_pos_pass_trainable:
                    yt = self._ff_convert_label(y_true, layer, FFConstants.POS)
                    losses.append(layer.ff_loss_fn(yt, y_pred))
        
        for layer, loss in zip(self.ff_pos_pass_trainable_layers, losses):
                gradients.append(tape.gradient(loss, layer.trainable_weights))
            
        del tape
        return gradients
        
    @tf.function
    def _ff_neg_pass_compute_gradients(self, X, y_true):
        losses = []
        gradients = []
        
        with tf.GradientTape(persistent=True) as tape:
            for layer, y_pred in zip(self.ff_layers, self(X, training=True)):
                if layer.ff_is_neg_pass_trainable:
                    yt = self._ff_convert_label(y_true, layer, FFConstants.NEG)
                    losses.append(layer.ff_loss_fn(yt, y_pred))
        
        for layer, loss in zip(self.ff_neg_pass_trainable_layers, losses):
                gradients.append(tape.gradient(loss, layer.trainable_weights))
            
        del tape
        return gradients
    
    @tf.function
    def _ff_pos_pass_update_params(self, gradients):
        for layer, grads in zip(self.ff_pos_pass_trainable_layers, gradients):
            layer.ff_optimizer.apply_gradients(
                zip(grads, layer.trainable_weights))
    
    @tf.function
    def _ff_neg_pass_update_params(self, gradients):
        for layer, grads in zip(self.ff_neg_pass_trainable_layers, gradients):
            layer.ff_optimizer.apply_gradients(
                zip(grads, layer.trainable_weights))
    
    ### Evaluation related methods

    def ff_reset_all_metrics(self):
        for layer in self.ff_layers:
            layer.ff_reset_metric()
    
    def ff_get_all_metric_results(self, results_dict):
        for layer in self.ff_layers:
            layer.ff_get_metric_results(results_dict)

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
            
    def ff_print_record(self, record):
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
    
    ### Utility methods
    
    @tf.function
    def _ff_convert_label(self, y_true, layer, _pass):
        if layer.ff_do_ff:
            if _pass == FFConstants.POS:
                return y_true * 0 + 1 # always 1
            elif _pass == FFConstants.NEG:
                return y_true * 0 # always 0
        else:
            return y_true