"""Data model for holding labelled data.

Supports basic data manipulation: normalisation, reduction, reconstruction,
class separation.
"""


import numpy as np

from .display import show_image


class Data(object):
    """Class for holding labelled data."""

    def __init__(self, raw_data, labels):
        """See class docstring."""
        self._raw_data = raw_data
        self._labels = labels
        self.reconstruct = lambda x: x
        self._reducer = None

    def __getitem__(self, index):
        """Wraps indexing the raw data."""
        return self._raw_data[index, :]

    def __len__(self):
        """Wraps the len() of the raw data."""
        return self._raw_data.shape[0]

    def get_label(self, index):
        """Get the label of the data at the given index."""
        return self._labels[index]

    def _set_reducer(self, reducer):
        self._reducer = reducer
        reducer.train(self._raw_data)

    reducer = property(fset=_set_reducer)

    @property
    def dimensions(self):
        """The number of dimensions the data has.  Read-only."""
        return self._raw_data.shape[1]

    @property
    def raw_data(self):
        """The raw data.  Read-only."""
        return self._raw_data.copy()

    @property
    def labels(self):
        """The labels by index.  Read-only."""
        return self._labels.copy()

    def normalise(self):
        """Normalise (l2) each vector in the data."""
        norms = np.linalg.norm(self._raw_data, axis=1)[:, np.newaxis]
        self._raw_data /= norms

    def reduce(self):
        """Apply the already-set dimensionality reducer to the data."""
        reduced_data = Data(raw_data=self._reducer.reduce(self._raw_data),
                            labels=self._labels)
        reduced_data.reconstruct = lambda x: self.reconstruct(
            self._reducer.reconstruct(x))
        return reduced_data

    def reconstruct_index(self, index):
        """Reconstruct the specified index with the parent data's reducer."""
        return self.reconstruct(self[index][1])

    def in_classes(self):
        """Get a dict of class labels containing the corresponding data."""
        class_labels = np.unique(self._labels)
        classes = {}
        for label in class_labels:
            selection_vector = self._labels == label
            class_data = self._raw_data[selection_vector, :]
            classes[label] = class_data
        return classes

    def first(self, n):
        """Take only the first n dimensions of the data."""
        return Data(raw_data=self._raw_data[:, :n], labels=self._labels)
