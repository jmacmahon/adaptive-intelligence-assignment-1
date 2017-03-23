import numpy as np

from .display import show_image

class Data(object):
    def __init__(self, raw_data, labels):
        self._raw_data = raw_data
        self._labels = labels
        self.reconstruct = lambda x: x
        self._reducer = None

    def __getitem__(self, index):
        return self._raw_data[index, :]

    def __len__(self):
        return self._raw_data.shape[0]

    def get_label(self, index):
        return self._labels[index]

    def _set_reducer(self, reducer):
        self._reducer = reducer
        reducer.train(self._raw_data)

    reducer = property(fset=_set_reducer)

    @property
    def dimensions(self):
        return self._raw_data.shape[1]

    @property
    def raw_data(self):
        return self._raw_data.copy()

    @property
    def labels(self):
        return self._labels.copy()

    def normalise(self):
        norms = np.linalg.norm(self._raw_data, axis=1)[:, np.newaxis]
        self._raw_data /= norms

    def reduce(self):
        reduced_data = Data(raw_data=self._reducer.reduce(self._raw_data),
                    labels=self._labels)
        reduced_data.reconstruct = lambda x: self.reconstruct(self._reducer.reconstruct(x))
        return reduced_data

    def reconstruct_index(self, index):
        return self.reconstruct(self[index][1])

    def in_classes(self):
        class_labels = np.unique(self._labels)
        classes = {}
        for label in class_labels:
            selection_vector = self._labels == label
            class_data = self._raw_data[selection_vector, :]
            classes[label] = class_data
        return classes
