import numpy as np

DEFAULT_NOISE_WEIGHT = 0.005
DEFAULT_LEARNING_RATE = 0.05

class SingleLayerCompetitiveNetwork(object):
    def __init__(self, inputs, outputs, learning_rate=DEFAULT_LEARNING_RATE,
                 noise_weight=DEFAULT_NOISE_WEIGHT):
        self._inputs = inputs
        self._outputs = outputs
        self._learning_rate = learning_rate
        self._noise_weight = noise_weight
        self._weights = np.random.rand(outputs, inputs)

    def train_one(self, data):
        output_firing_rate = np.dot(self._weights, data)
        noise = self._noise_weight * np.random.rand(self._outputs)

        # winner_output = np.max(output_firing_rate + noise)
        winner_index = np.argmax(output_firing_rate + noise)

        dw = self._learning_rate * (data - self._weights[winner_index, :])

        self._weights[winner_index, :] += dw

        return winner_index, dw, self._weights

    def train_many(self, data):
        for input_ in data:
            yield self.train_one(input_)

    def classify_many(self, data):
        output_firing_rate = np.dot(self._weights, data.T)
        winners = np.argmax(output_firing_rate, axis=0)
        return winners

    def evaluate(self, data, labels):
        pass
