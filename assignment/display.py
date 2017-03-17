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
        # TODO calculate size here instead of 2x2 hardcoded
        ax = fig.add_subplot(subplots_v, subplots_h, i, projection='3d')
        legend_data = []
        for (label, class_data) in classes.items():
            scatter = ax.scatter(class_data[:, x], class_data[:, y], class_data[:, z])
            legend_data.append((scatter, label))
        ax.legend(*zip(*legend_data))
        ax.set_title('Dimensions: {}, {}, {}'.format(x, y, z))
    plt.tight_layout()
    plt.show()
