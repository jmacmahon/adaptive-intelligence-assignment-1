import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from pickle import load

from assignment.load import load_data_pickle
from assignment.display import (get_3d_classes_figures, get_3d_tunings_figures,
                                create_weights_plot)
from assignment.dimensionality import PCAReducer
from assignment.evaluation import fuzz_evaluate
from assignment.neural import SingleLayerCompetitiveNetwork
from assignment.util import random_iter, consume

data = load_data_pickle()
data.normalise()


def pca_scatter_graphs():
    data.reducer = PCAReducer(5)
    reduced = data.reduce()

    # Separate data into a dict with class labels as its keys
    in_classes = reduced.in_classes()
    figures = get_3d_classes_figures(in_classes)

    # For saving or viewing from IPython
    return figures


def compute_performance_data(fast=False):
    # Slow version takes about 3 hours on my machine; fast version about 1
    # minute, but the data is mostly useless.

    if not fast:
        iterations = 15000
        params = {
            'learning_rate': np.power(1000, np.arange(-1, 0.1, 0.1)),
            'learning_rate_decay': np.power(1000, np.arange(-1, 0.1, 0.1)),
            'outputs': np.arange(10, 20, 1)
        }
    else:
        iterations = 5000
        params = {
            'learning_rate': np.power(1000, np.arange(-1, 0, 0.5)),
            'learning_rate_decay': np.power(1000, np.arange(-1, 0, 0.5)),
            'outputs': np.arange(10, 20, 5)
        }

    # This will take a while.
    performance_data = fuzz_evaluate(
        partial(SingleLayerCompetitiveNetwork, data.dimensions), params, data,
        iterations)
    return performance_data


def load_performance_data_from_pickle():
    with open('performance_data_cached.pickle', 'rb') as f:
        return load(f)


def performance_surface_graphs(from_pickle=True, fast=False):
    if from_pickle:
        performance_data = load_performance_data_from_pickle()
    else:
        performance_data = compute_performance_data(fast)

    figures = get_3d_tunings_figures(performance_data[1],
                                     ('Learning rate',
                                      'Learning rate decay',
                                      'Outputs'))
    return figures


def weight_change_graph():
    learning_rate = 0.032
    learning_rate_decay = 0.002
    outputs = 15
    iterations = 15000

    nn = SingleLayerCompetitiveNetwork(28*28, outputs,
                                       learning_rate=learning_rate,
                                       learning_rate_decay=learning_rate_decay)
    training_data_iter = random_iter(data.raw_data, iterations)
    training_iter = nn.train_many(training_data_iter)

    weight_changes = [weight_change for _, _, weight_change in training_iter]

    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.semilogx(weight_changes)

    return fig


def units_and_covariance_graph():
    learning_rate = 0.032
    learning_rate_decay = 0.002
    outputs = 15
    iterations = 2**16

    nn = SingleLayerCompetitiveNetwork(28*28, outputs,
                                       learning_rate=learning_rate,
                                       learning_rate_decay=learning_rate_decay)
    training_data_iter = random_iter(data.raw_data, iterations)
    consume(nn.train_many(training_data_iter))

    show_units = create_weights_plot(outputs)
    units_figure = show_units(nn._weights)

    covariance_figure = plt.figure()
    covariance_axes = covariance_figure.add_subplot(111)
    covariance = np.cov(nn._weights)
    covariance_axes.imshow(covariance)

    return (units_figure, covariance_figure)
