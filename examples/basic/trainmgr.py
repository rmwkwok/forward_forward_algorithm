import tensorflow as tf
from ffobjects import BaseFFLayer

class TrainMgr:
    def __init__(self, layers, train_seq):
        self.layers = layers
        self.train_seq = train_seq
        self.history = []
        self.tf_train_fns = dict()
        self.metric_monitor_buffer = dict()
        self._get_all_metrics()
        '''
        A training manager for the training process and metrics monitor.
        When calls the `ff_train()` method, it runs 3 layers of loop.
        The outermost loops over `trainable_layers_list` in which each
        element contains the a list of trainable layers and `epochs`.
        The middle loop iterates through the epochs. The innermost takes
        one dataset out at a time and carry the task associated to the
        dataset. 
        Since different tasks requires different operation from the
        same layer, to exploit tensorflow graph while avoid retracing,
        the training manager keeps a dictionary of these graphed 
        functions so that one graph serves one configuration of tasks,
        and the stored graphs may be reuse for the same config of tasks.
        
        Args:
            layers: a `dict` of FFLayers and / or tensorflow layers.
            train_seq: a python function that connects the `layers` up.
        '''
        
    def ff_train(self, datasets, trainable_layers_list, show_metrics_max=[]):
        '''
        Training the layers following the `train_seq`. 
        
        Args:
            datasets: a `list` of `(task, tf.data.Dataset)`. `task` can
                take any one of `TASK_TRANSFORM`, `TASK_TRAIN_POS`
                `TASK_TRAIN_NEG`, `TASK_EVAL`, `TASK_EVAL_DUPED` defined
                in class `BaseFFLayer`.
            trainable_layers_list: a `list` of 
                `[[layer_1, layer_2, ...], epochs]`. Each listing 
                element describes which layers are trainable for 
                `epochs` round of training. The layer name `layer_1`
                should reference back to the name in the `layers` 
                dictionary that has been passed in when instaniating 
                this object.
            show_metrics_max: a `list` of `str`. Listed layer's maximum
                metric value among epochs will be shown in the training
                progress bar.  The layer name should reference back to 
                the name in the `layers` dictionary that has been passed
                in when instaniating this object.
                
        Returns:
            self.
        '''
        ff_layers = {n: l for n, l in self.layers.items() 
                              if isinstance(l, BaseFFLayer)}
        
        for trainable_layers, epochs in trainable_layers_list:
            self._init_metric_monitor(
                show_metrics_max, trainable_layers, epochs)
            for epoch in range(epochs):
                self._reset_metric_objects()
                for task, dataset in datasets:
                    for name, layer in ff_layers.items():
                        layer.ff_set_task(
                            task if name in trainable_layers else\
                            BaseFFLayer.TASK_TRANSFORM)
                    key = (task, *trainable_layers, str(dataset.element_spec))
                    signature = tf.data.DatasetSpec(dataset.element_spec)
                    self.tf_train_fns\
                        .setdefault(key, self._get_new_fn(signature))(dataset)
                self._update_metrics(epoch)
            self._save_history_buffer()
        return self
    
    def _get_new_fn(self, dataset_signature):
        @tf.function(input_signature=[dataset_signature])
        def _train_fn(dataset):
            for X, y, y_ff in dataset:
                self.train_seq(X, y, y_ff)
        return _train_fn
        
    # Utilities: Metric related
    def _get_all_metrics(self):
        self._metrics = {}
        for name, layer in self.layers.items():
            if isinstance(layer, BaseFFLayer):
                if isinstance(layer.ff_metric, tf.keras.metrics.Metric):
                    self._metrics[name] = layer.ff_metric
                if isinstance(layer.ff_metric_duped, tf.keras.metrics.Metric):
                    self._metrics[f'{name}_duped'] = layer.ff_metric_duped
        
    def _init_metric_monitor(self, show_metrics_max, trainable_layers, epochs):
        self._monitoring_metrics = {
            name: metric for name, metric in self._metrics.items() \
                if name.rstrip('_duped') in trainable_layers}
        
        self._best_metric_buffer = {
            f'best_{name}': -9999999 
                for name, v in self._monitoring_metrics.items() \
                    if name.rstrip('_duped') in show_metrics_max}
        
        self._hist_buffer = dict(
            trainable_layers=trainable_layers,
            **{n: [] for n in self._monitoring_metrics})
        
        pbar_names = list(self._monitoring_metrics) +\
                     list(self._best_metric_buffer)
        self._pbar = tf.keras.utils.Progbar(epochs, 
                                            stateful_metrics=pbar_names)
        
    def _reset_metric_objects(self):
        [metric.reset_state() for metric in self._monitoring_metrics.values()]
        
    def _update_metrics(self, epoch):
        pbar_metric = []
        for name, metric in self._monitoring_metrics.items():
            v = metric.result()
            self._hist_buffer[name].append(v)
            pbar_metric.append((name, v))
        for name, v in self._best_metric_buffer.items():
            new_v = max(v, self._hist_buffer[name[5:]][-1])
            self._best_metric_buffer[name] = new_v
            pbar_metric.append((name, self._best_metric_buffer[name]))
        self._pbar.update(epoch+1, pbar_metric)
        
    def _save_history_buffer(self):
        temp = {k: v if k not in self._monitoring_metrics or len(v) == 0 else\
                   tf.concat(v, 0).numpy() 
                    for k, v in self._hist_buffer.items() }
        self.history.append(temp)