"""A suite of feature selectors and dimensionality reducers."""

import numpy as np
from scipy.linalg import eigh
from scipy.stats import f_oneway
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


class DropFirstNSelector(IdentityReducer):
    """Feature selector which drops the first n features."""

    def __init__(self, n=1):
        """See class docstring."""
        self._n = n
        start_dim = "k"
        end_dim = "(k-{})".format(n)
        getLogger('assignment.dimensionality.dropfirstn')\
            .info("Init Drop First N feature selector ({} -> {} dimensions)"
                  .format(start_dim, end_dim))

    def reduce(self, data):
        """Drop the first n features from the input vector(s).

        :param data: The data to be reduced

        :return: The reduced feature vector(s)
        """
        return data.transpose()[self._n:].transpose()


class BestKSelector(IdentityReducer):
    """Select the 'best' k features according to their ANOVA f-value."""

    def __init__(self, k):
        """See class docstring."""
        self._k = k

    def train(self, train_data, labels):
        """Train the selector -- find the 'best' features.

        :param train_data: The n training vectors
        :param labels: The n class labels corresponding to the vectors in
            train_data
        """
        classes = [train_data[labels == label] for label in np.unique(labels)]
        # Use f_oneway from scipy as a divergence measure
        scores = f_oneway(*classes).statistic
        self._best_k = np.argsort(scores)[::-1][:self._k]
        getLogger('assignment.dimensionality.bestk')\
            .info("Trained Best-K feature selector")

    def reduce(self, data):
        """Select the best k features from the input vector(s).

        :param data: The data to be reduced
        """
        getLogger('assignment.dimensionality.bestk')\
            .debug("Best-K reduced some samples")
        return data.transpose()[self._best_k].transpose()


class BorderTrimReducer(IdentityReducer):
    """Feature selector which trims the border of the image vector.

    When the feature vector represents an image, often the borders will contain
    useless or noisy data which, when trimmed off, can improve the classifier.

    :param top, bottom, left, right: The number of pixels to trim off the top,
        bottom, left and right of the image.
    :param startshape: The initial shape of the image in pixels.
    """

    def __init__(self, top=10, bottom=10, left=10, right=10,
                 startshape=(30, 30)):
        """See class docstring."""
        self._top = top
        self._bottom = bottom
        self._right = right
        self._left = left
        self._startshape = startshape
        start_dim = startshape[0] * startshape[1]
        end_dim = ((startshape[0] - top - bottom) *
                   (startshape[1] - left - right))
        getLogger('assignment.dimensionality.bordertrim')\
            .info("Init Border Trim Reducer ({} -> {} dimensions)"
                  .format(start_dim, end_dim))

    def reduce(self, data, flatten=True):
        """Trim the input image vector(s).

        :param data: The input data to be trimmed
        :param flatten: (Default = True) Whether to flatten the output vector
        """
        data = data.reshape(-1, *self._startshape)
        height = self._startshape[0]
        width = self._startshape[1]
        out = data[:,
                   self._left:(width - self._right),
                   self._top:(height - self._bottom)]
        getLogger('assignment.dimensionality.bordertrim')\
            .debug("Trimmed border")
        if not flatten:
            return out
        if data.shape[0] == 1:
            return out.flatten()
        new_len = ((width - self._left - self._right) *
                   (height - self._top - self._bottom))
        return out.reshape(-1, new_len)
