"""A collection of helper display functions using a pyplot backend."""

import matplotlib.pyplot as plt
from matplotlib.cm import gray
from math import sqrt, ceil, floor
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations


def show_image(pixels, shape):
    """Display the provided pixels in the provided shape."""
    shaped_matrix = np.reshape(pixels, shape, order='F')
    image = plt.imshow(shaped_matrix, cmap=gray)
    return image


def show_3d_classes(classes):
    dimensions = next(iter(classes.values())).shape[1]
    fig = plt.figure()
    i = 0

    # r = 3 for 3-dimensions
    combinations_ = list(combinations(range(dimensions), 3))
    subplots_h = floor(sqrt(len(combinations_)))
    subplots_v = ceil(len(combinations_) / subplots_h)

    for (x, y, z) in combinations_:
        i += 1
        ax = fig.add_subplot(subplots_v, subplots_h, i, projection='3d')
        legend_data = []
        for (label, class_data) in classes.items():
            scatter = ax.scatter(class_data[:, x], class_data[:, y], class_data[:, z])
            legend_data.append((scatter, label))
        ax.legend(*zip(*legend_data))
        ax.set_title('Dimensions: {}, {}, {}'.format(x, y, z))
    plt.tight_layout()
    plt.show()


def create_weights_plot(width, height):
    plt.ion()
    fig, axes = plt.subplots(height, width)
    plt.show()
    def show_units(units):
        for i in range(height):
            for j in range(width):
                if 5 * i + j < units.shape[0]:
                    output_neuron = units[5 * i + j, :].reshape((28,28),order = 'F')
                    axes[i, j].clear()
                    axes[i, j].imshow(output_neuron, interpolation='nearest')
                axes[i, j].get_xaxis().set_ticks([])
                axes[i, j].get_yaxis().set_ticks([])
        plt.draw()
        plt.pause(0.0001)

    return show_units
