from .display import show_image

class Data(object):
    def __init__(self, raw_data, labels):
        self._raw_data = raw_data
        self._labels = labels

    def get_train_letter(self, index):
        return Letter(data=self._raw_data[index, :],
                      label=self._labels[index],
                      index=index)

    def __iter__(self):
        for i in range(self._raw_data.shape[0]):
            yield self.get_train_letter(i)


class Letter(object):
    def __init__(self, data, label=None, shape=(28, 28), index=None):
        self.data = data
        self.label = label
        self._shape = shape
        self._index = index

    def show(self):
        show_image(self.data, self._shape)

    def __repr__(self):
        return 'Letter<index: {}, label: {}>'.format(
            repr(self._index), repr(self.label))
