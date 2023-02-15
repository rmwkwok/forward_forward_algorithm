import random
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

def set_seed(seed=100):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
def plot_training_curves(train_mgr):
    for _hist in train_mgr.history:
        metriced_layers = [l for l in _hist if l != 'trainable_layers']

        fig, axes = plt.subplots(1, 5, figsize=(20,3))
        fig.suptitle('trainable_layers:' + str(_hist['trainable_layers']))
        for layer, ax in zip(metriced_layers, axes.flatten()):
            ax.set_title(layer)
            ax.set_xlabel('epoch')
            ax.set_ylabel('metric')
            ax.plot(_hist[layer])
        plt.tight_layout()
        plt.show()