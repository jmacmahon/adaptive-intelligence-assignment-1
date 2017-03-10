from .display import show_image

class Data(object):
    def __init__(self, raw_data, labels):
        self._raw_data = raw_data
        self._labels = labels

    def __getitem__(self, index):
        return (self._labels[index], self._raw_data[index, :])

    def _set_reducer(self, reducer):
        self._reducer = reducer
        reducer.train(self._raw_data)

    reducer = property(fset=_set_reducer)

    def reduce(self):
        return Data(raw_data=self._reducer.reduce(self._raw_data),
                    labels=self._labels)

    def in_classes(self):
        #TODO
        pass
