"""A suite of feature selectors and dimensionality reducers.

:note: this code was copied from my COM3004 assignment and extraneous classes
removed.
"""

import numpy as np
from scipy.linalg import eigh
from logging import getLogger


class IdentityReducer(object):
    """Base class for a dimensionality reducer."""

    def train(self, *_, **__):
        """'Train' the reducer (in this case do nothing).

        :note: This method is provided for subclassing purposes -- some
            reducers may not need to be trained but must have a `train` method
            which accepts arguments.
        """
        pass


class PCAReducer(IdentityReducer):
    """Principle Component Analysis (PCA) dimensionality reducer.

    :param n: The number of dimensions (eigenvectors) to reduce to.
    """

    def __init__(self, n=40):
        """See class docstring."""
        self._n = n

    def train(self, train_data, *_, **__):
        """Train the PCA reducer.

        :param train_data: The training data to train on
        """
        cov = np.cov(train_data, rowvar=0)
        dim = cov.shape[0]
        _, eigenvectors = eigh(cov, eigvals=(dim - self._n, dim - 1))
        eigenvectors = np.fliplr(eigenvectors)
        self._eigenvectors = eigenvectors

        self._mean = np.mean(train_data, axis=0)
        getLogger('assignment.dimensionality.pca')\
            .info("Trained PCA reducer ({} -> {} dimensions)"
                  .format(dim, self._n))

    def reduce(self, data):
        """Reduce the provided data.

        :param data: The data to be reduced

        :return: The PCA-reduced feature vector(s)
        """
        centred_data = data - self._mean
        vs = self._eigenvectors[:, :self._n]
        getLogger('assignment.dimensionality.pca')\
            .debug("PCA-reduced some samples")
        return np.dot(centred_data, vs)

    def reconstruct(self, data):
        """Reconstruct the original data from the provided PCA-vector."""
        return np.dot(data, self._eigenvectors.T)
