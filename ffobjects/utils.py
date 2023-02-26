import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def set_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
class Plotter:
    def __init__(self, plots_per_row, figsize):
        self.axes = []
        self.figsize = figsize
        self.plots_per_row = plots_per_row
        self.gen_axes()
        
    def get_ax(self):
        if len(self.axes) == 0:
            self.close()
            self.gen_axes()
        return self.axes.pop(0)
        
    def gen_axes(self):
        fig, axes = plt.subplots(1, self.plots_per_row, figsize=self.figsize)
        self.fig = fig
        self.axes.extend(axes)
        
    def close(self):
        plt.tight_layout()
        plt.show()

def plot_training_curves(train_mgr):
    for _hist in train_mgr.history:
        metriced_layers = [l for l in _hist if l != 'trainable_layers']

        plotter = Plotter(5, (20, 3))
        plotter.fig.suptitle('trainable_layers:' + str(_hist['trainable_layers']))
        for layer in metriced_layers:
            ax = plotter.get_ax()
            ax.set_title(layer)
            ax.set_xlabel('epoch')
            ax.set_ylabel('metric')
            ax.plot(_hist[layer])
        plotter.close()

# Routes related for FFRoutedDense
def _ones_partial(indices, length):
    return tf.reduce_sum(tf.one_hot(indices, length), axis=0, keepdims=True)

def _route_classes_to_units(classes, num_classes, units, num_units):
    a = _ones_partial(classes, num_classes)
    b = _ones_partial(units, num_units)
    return tf.transpose(a) @ b

def _get_route(num_classes, num_units, classes_to_units_list):
    '''
    Args:
        num_classes: `int`. total number of classes
        num_units: `int`.total number of FFDense units
        classes_to_units_list: `list` of `tuple` `(classes, units)`.
            `classes` is a `Tensor` of classes' numbers that are to be
            routed to `units` which is a `Tensors` of units' numbers.
            e.g. (tf.range(0,  5), tf.range( 0,  50)) maps classes 0-4
            to units 0-49.
    Returns:
        `Tensor` of shape (`num_classes`, `num_units`) representing
            which class is routed to which unit.
    '''
    return tf.math.add_n([
               _route_classes_to_units(classes, num_classes, units, num_units)
                   for classes, units in classes_to_units_list])

def get_routes(num_classes, num_units, num_routes, seed=1,
               split_mode='NoSplitting'):
    '''
    Given `num_classes` and `num_units`, generate `num_routes` routes
        based on the criteria set by `classes_to_units_list`.
    `num_routes` usually equals to the number of hidden layer that need
        a route.
    `mode` is one of `['NoSplitting', 'RandomSplitting',
        'SameSplitting']`
    '''
    classes = tf.range(num_classes)
    classes = tf.random.shuffle(classes, seed=seed)

    for i in range(num_routes):
        if split_mode == 'NoSplitting':
            classes_to_units_list =  [
                (classes, tf.range(0,  num_units)),
            ]
        else:
            classes_to_units_list =  [
                (classes[:5], tf.range(0,  num_units//2)),
                (classes[5:], tf.range(num_units//2, num_units)),
            ]
        yield _get_route(num_classes, num_units, classes_to_units_list)

        if split_mode == 'RandomSplitting':
            classes = tf.random.shuffle(classes, seed=seed)
