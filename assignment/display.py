"""A collection of helper display functions using a pyplot backend."""

import matplotlib.pyplot as plt
from matplotlib.cm import gray
from math import sqrt, floor
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def show_image(pixels, shape):
    """Display the provided pixels in the provided shape."""
    shaped_matrix = np.reshape(pixels, shape, order='F')
    image = plt.imshow(shaped_matrix, cmap=gray)
    return image


def show_3d_classes(classes):
    pass
