import numpy as np

from .display import show_image

class Data(object):
    def __init__(self, raw_data, labels):
        self._raw_data = raw_data
        self._labels = labels
        self._reconstructor = None
        self._reducer = None

    def __getitem__(self, index):
        return (self._labels[index], self._raw_data[index, :])

    def _set_reducer(self, reducer):
        self._reducer = reducer
        reducer.train(self._raw_data)

    reducer = property(fset=_set_reducer)

    def reduce(self):
        reduced_data = Data(raw_data=self._reducer.reduce(self._raw_data),
                    labels=self._labels)
        reduced_data._reconstructor = self._reducer.reconstruct
        return reduced_data

    def reconstruct(self, index):
        return self._reconstructor(self[index][1])

    def in_classes(self):
        class_labels = np.unique(self._labels)
        classes = {}
        for label in class_labels:
            selection_vector = self._labels == label
            class_data = self._raw_data[selection_vector, :]
            classes[label] = class_data
        return classes
