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