"""A collection of helper display functions using a pyplot backend."""

import matplotlib.pyplot as plt
from matplotlib.cm import gray
from math import sqrt, floor
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
    for (x, y, z) in combinations(range(dimensions), 3):
        # r = 3 for 3-dimensions
        i += 1
        # TODO calculate size here instead of 2x2 hardcoded
        ax = fig.add_subplot(2, 2, i, projection='3d')
        for (label, class_data) in classes.items():
            ax.scatter(class_data[:, x], class_data[:, y], class_data[:, z])
    plt.show()
