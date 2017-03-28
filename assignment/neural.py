import numpy as np

DEFAULT_NOISE_WEIGHT = 0.005
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_LEARNING_RATE_DECAY = 0
DEFAULT_ALPHA = 0.5


class EvaluableClassifier(object):
    def evaluate(self, data, labels):
        classified = self.classify_many(data)

        # A confusion matrix is a table of true label vs. predicted label.
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
                 learning_rate_decay=DEFAULT_LEARNING_RATE_DECAY,
                 noise_weight=DEFAULT_NOISE_WEIGHT, alpha=DEFAULT_ALPHA):
        self._inputs = inputs
        self._outputs = outputs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._noise_weight = noise_weight
        # Initialise the weights randomly for symmetry-breaking
        self._weights = np.random.rand(outputs, inputs)
        self._t = 0
        self._average_dw = None
        self._alpha = alpha

    def train_one(self, data):
        self._t += 1

        output_firing_rate = np.dot(self._weights, data)
        noise = self._noise_weight * np.random.rand(self._outputs)

        # Add the noise to the firing rate and take the neuron with the largest
        # input
        winner_index = np.argmax(output_firing_rate + noise)

        eta = self._learning_rate * self._t ** (-self._learning_rate_decay)
        dw = eta * (data - self._weights[winner_index, :])

        self._weights[winner_index, :] += dw

        dw_magnitude = np.dot(dw, dw.T)

        if self._average_dw is None:
            self._average_dw = dw_magnitude
        else:
            self._average_dw = (self._alpha * dw_magnitude +
                                (1 - self._alpha) * self._average_dw)

        return winner_index, self._weights, self._average_dw

    def train_many(self, data):
        return map(self.train_one, data)

    def classify_many(self, data):
        output_firing_rate = np.dot(self._weights, data.T)
        winners = np.argmax(output_firing_rate, axis=0)
        return winners

class TwoLayerCompetitiveNetwork(EvaluableClassifier):
    def __init__(self, inputs, outputs, intermediates, groups,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 learning_rate_decay=DEFAULT_LEARNING_RATE_DECAY,
                 noise_weight=DEFAULT_NOISE_WEIGHT):
        self._inputs = inputs
        self._outputs = outputs
        self._intermediates = intermediates
        self._groups = groups
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._noise_weight = noise_weight
        self._weights_l1 = np.random.rand(intermediates, inputs)
        self._weights_l2 = np.random.rand(outputs, intermediates)
        self._t = 0

    def train_one(self, data):
        self._t += 1
        eta = self._learning_rate * self._t ** (-self._learning_rate_decay)

        l1_noise = self._noise_weight * np.random.rand(self._intermediates)
        l1_input_activity = np.dot(self._weights_l1, data) + l1_noise

        group_size = int(self._intermediates / self._groups)
        l1_firing_rate = np.zeros(self._intermediates)
        for group_index in range(self._groups):
            group = l1_input_activity[group_index * group_size
                                      :(group_index + 1) * group_size]
            group_winner_index = np.argmax(group) + group_index * group_size
            l1_firing_rate[group_winner_index] = 1

            group_dw = (eta * (data - self._weights_l1[group_winner_index, :]))
            self._weights_l1[group_winner_index, :] += group_dw

        l2_noise = self._noise_weight * np.random.rand(self._outputs)
        l2_input_activity = np.dot(self._weights_l2, l1_firing_rate) + l2_noise

        l2_winner_index = np.argmax(l2_input_activity)

        # Possibly use l1_input_activity here instead of l1_firing_rate
        l2_dw = (eta * (l1_firing_rate - self._weights_l2[l2_winner_index, :]))
        self._weights_l2[l2_winner_index, :] +=  l2_dw

        return l2_winner_index, self._weights_l1, self._weights_l2

    def train_many(self, data):
        return map(self.train_one, data)

    def classify_many(self, data):
        l1_input_activity = np.dot(self._weights_l1, data.T)
        group_size = int(self._intermediates / self._groups)
        l1_firing_rate = np.zeros(l1_input_activity.shape)
        for group_index in range(self._groups):
            group = l1_input_activity[group_index * group_size
                                      :(group_index + 1) * group_size, :]
            group_winner_indices = np.argmax(group, axis=0) + group_index * group_size
            for i in range(group_winner_indices.shape[0]):
                winner_index = group_winner_indices[i]
                l1_firing_rate[winner_index, i] = 1

        l2_input_activity = np.dot(self._weights_l2, l1_firing_rate)
        winners = np.argmax(l2_input_activity, axis=0)
        return winners
