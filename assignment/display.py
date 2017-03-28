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
        ax.legend(*zip(*legend_data), loc=2)
        ax.set_xlabel('Dim #{}'.format(x))
        ax.set_ylabel('Dim #{}'.format(y))
        ax.set_zlabel('Dim #{}'.format(z))
        ax.set_title('Dimensions: {}, {}, {}'.format(x, y, z))
    plt.tight_layout()
    plt.show()


def show_3d_tunings(tunings, labels=None):
    parameters = tunings.shape[1] - 1
    combinations_ = list(combinations(range(parameters), r=2))

    fig = plt.figure()
    subplots_v = floor(sqrt(len(combinations_)))
    subplots_h = ceil(len(combinations_) / subplots_v)

    i = 0
    for x_index, y_index in combinations_:
        i += 1
        x_values = np.unique(tunings[:, x_index])
        y_values = np.unique(tunings[:, y_index])
        grid = np.meshgrid(x_values, y_values)
        z_values_array = []
        for x_value, y_value in zip(grid[0].reshape(-1), grid[1].reshape(-1)):
            selector = np.all([tunings[:, x_index] == x_value,
                               tunings[:, y_index] == y_value], axis=0)
            z_value = np.mean(tunings[selector, parameters])
            z_values_array.append(z_value)
        z_values = np.array(z_values_array).reshape(grid[0].shape)

        ax = fig.add_subplot(subplots_v, subplots_h, i, projection='3d')
        if labels is not None:
            ax.set_xlabel(labels[x_index])
            ax.set_ylabel(labels[y_index])
        ax.plot_surface(*grid, z_values)
    plt.tight_layout()
    plt.show()

def get_3d_tunings_figures(tunings, labels=None):
    parameters = tunings.shape[1] - 1
    combinations_ = list(combinations(range(parameters), r=2))

    figures = {}
    for x_index, y_index in combinations_:
        x_values = np.unique(tunings[:, x_index])
        y_values = np.unique(tunings[:, y_index])
        grid = np.meshgrid(x_values, y_values)
        z_values_array = []
        for x_value, y_value in zip(grid[0].reshape(-1), grid[1].reshape(-1)):
            selector = np.all([tunings[:, x_index] == x_value,
                               tunings[:, y_index] == y_value], axis=0)
            z_value = np.mean(tunings[selector, parameters])
            z_values_array.append(z_value)
        z_values = np.array(z_values_array).reshape(grid[0].shape)

        fig = plt.figure()
        figures[(x_index, y_index)] = fig
        ax = fig.add_subplot(111, projection='3d')
        if labels is not None:
            ax.set_xlabel(labels[x_index])
            ax.set_ylabel(labels[y_index])
        ax.plot_surface(*grid, z_values)
    return figures

def get_3d_classes_figures(classes):
    dimensions = next(iter(classes.values())).shape[1]
    combinations_ = list(combinations(range(dimensions), 3))
    figures = {}
    for (x, y, z) in combinations_:
        fig = plt.figure()
        figures[(x, y, z)] = fig
        ax = fig.add_subplot(111, projection='3d')
        legend_data = []
        for (label, class_data) in classes.items():
            scatter = ax.scatter(class_data[:, x], class_data[:, y], class_data[:, z])
            legend_data.append((scatter, label))
        ax.legend(*zip(*legend_data), loc=2)
        ax.set_xlabel('Dim #{}'.format(x))
        ax.set_ylabel('Dim #{}'.format(y))
        ax.set_zlabel('Dim #{}'.format(z))
        ax.set_title('Dimensions: {}, {}, {}'.format(x, y, z))
    return figures


def create_weights_plot(weights_number):
    height = floor(sqrt(weights_number))
    width = ceil(weights_number / height)

    plt.ion()
    fig, axes = plt.subplots(height, width)
    plt.show()
    def show_units(units):
        for i in range(height):
            for j in range(width):
                if width * i + j < units.shape[0]:
                    output_neuron = units[width * i + j, :].reshape((28,28),order = 'F')
                    axes[i, j].clear()
                    axes[i, j].imshow(output_neuron, interpolation='nearest')
                axes[i, j].get_xaxis().set_ticks([])
                axes[i, j].get_yaxis().set_ticks([])
        plt.draw()
        plt.pause(0.0001)
        return fig, axes

    return show_units
