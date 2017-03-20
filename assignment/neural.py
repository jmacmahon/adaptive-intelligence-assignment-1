import numpy as np

DEFAULT_NOISE_WEIGHT = 0.005
DEFAULT_LEARNING_RATE = 0.05


class EvaluableClassifier(object):
    def evaluate(self, data, labels):
        classified = self.classify_many(data)

        confusion_matrix = np.zeros((self._inputs, self._outputs))
        for true_label, predicted_label in zip(labels, classified):
            confusion_matrix[int(true_label), int(predicted_label)] += 1

        # We evaluate each cluster's label by majority rule
        correspondence = np.argmax(confusion_matrix, axis=0)

        correct_results = 0
        for cluster_index, cluster_label in zip(
            range(self._outputs), correspondence):
            correct_results += confusion_matrix[cluster_label, cluster_index]

        return correct_results / np.sum(confusion_matrix)


class SingleLayerCompetitiveNetwork(EvaluableClassifier):
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
        return map(self.train_one, data)

    def classify_many(self, data):
        output_firing_rate = np.dot(self._weights, data.T)
        winners = np.argmax(output_firing_rate, axis=0)
        return winners
