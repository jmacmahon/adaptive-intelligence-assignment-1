"""All the neural network code.

As well as a single layer competitive network, there is also code for a
two-layer network.  This was just for interest and I do not expect to be
assessed on it.
"""

import numpy as np

DEFAULT_NOISE_WEIGHT = 0.005
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_LEARNING_RATE_DECAY = 0
DEFAULT_AVERAGE_ALPHA = 0.5


class EvaluableClassifier(object):
    """A mixin evaluation method class."""

    def evaluate(self, data, labels):
        """Perform a classification evaluation of the provided data and labels.

        :note: Guesses the label for each cluster based on the majority of
            points' labels in that cluster.
        """
        classified = self.classify_many(data)
        number_of_labels = np.unique(labels).shape[0]

        # A confusion matrix is a table of true label vs. predicted label.
        confusion_matrix = np.zeros((number_of_labels, self._outputs))
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
    """A single layer competitive neural network.

    :param inputs: The number of input neurons, equal to the dimensionality of
        the data.

    :param outputs: The number of output units.

    :param learning_rate: The learning rate (eta).

    :param learning_rate_decay: The learning rate decay, as outlined in the
        report and in Hertz 1991.

    :param noise_weight: The weighting given to the noise smear.

    :param average_alpha: The 0-1 parameter of the moving average for
        change-in-weight calculation.
    """

    def __init__(self, inputs, outputs, learning_rate=DEFAULT_LEARNING_RATE,
                 learning_rate_decay=DEFAULT_LEARNING_RATE_DECAY,
                 noise_weight=DEFAULT_NOISE_WEIGHT,
                 average_alpha=DEFAULT_AVERAGE_ALPHA):
        """See class docstring."""
        self._inputs = inputs
        self._outputs = outputs
        self._learning_rate = learning_rate
        self._learning_rate_decay = learning_rate_decay
        self._noise_weight = noise_weight
        # Initialise the weights randomly for symmetry-breaking
        self._weights = np.random.rand(outputs, inputs)
        self._t = 0
        self._average_dw = None
        self._average_alpha = average_alpha

    def train_one(self, data):
        """Perform training on the network using a single data sample."""
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
            self._average_dw = (self._average_alpha * dw_magnitude +
                                (1 - self._average_alpha) * self._average_dw)

        return winner_index, self._weights, self._average_dw

    def train_many(self, data):
        """Get an iterable training the network over every item in `data`."""
        return map(self.train_one, data)

    def classify_many(self, data):
        """Perform classification in bulk on multiple input vectors."""
        output_firing_rate = np.dot(self._weights, data.T)
        winners = np.argmax(output_firing_rate, axis=0)
        return winners


class TwoLayerCompetitiveNetwork(EvaluableClassifier):
    """A two-layer competitive neural network."""

    def __init__(self, inputs, outputs, intermediates, groups,
                 learning_rate=DEFAULT_LEARNING_RATE,
                 learning_rate_decay=DEFAULT_LEARNING_RATE_DECAY,
                 noise_weight=DEFAULT_NOISE_WEIGHT):
        """See class docstring."""
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
        """Perform training on the network using a single data sample."""
        self._t += 1
        eta = self._learning_rate * self._t ** (-self._learning_rate_decay)

        l1_noise = self._noise_weight * np.random.rand(self._intermediates)
        l1_input_activity = np.dot(self._weights_l1, data) + l1_noise

        group_size = int(self._intermediates / self._groups)
        l1_firing_rate = np.zeros(self._intermediates)
        for group_index in range(self._groups):
            group = l1_input_activity[group_index * group_size:
                                      (group_index + 1) * group_size]
            group_winner_index = np.argmax(group) + group_index * group_size
            l1_firing_rate[group_winner_index] = 1

            group_dw = (eta * (data - self._weights_l1[group_winner_index, :]))
            self._weights_l1[group_winner_index, :] += group_dw

        l2_noise = self._noise_weight * np.random.rand(self._outputs)
        l2_input_activity = np.dot(self._weights_l2, l1_firing_rate) + l2_noise

        l2_winner_index = np.argmax(l2_input_activity)

        # Possibly use l1_input_activity here instead of l1_firing_rate
        l2_dw = (eta * (l1_firing_rate - self._weights_l2[l2_winner_index, :]))
        self._weights_l2[l2_winner_index, :] += l2_dw

        return l2_winner_index, self._weights_l1, self._weights_l2

    def train_many(self, data):
        """Get an iterable training the network over every item in `data`."""
        return map(self.train_one, data)

    def classify_many(self, data):
        """Perform classification in bulk on multiple input vectors."""
        l1_input_activity = np.dot(self._weights_l1, data.T)
        group_size = int(self._intermediates / self._groups)
        l1_firing_rate = np.zeros(l1_input_activity.shape)
        for group_index in range(self._groups):
            group = l1_input_activity[group_index * group_size:
                                      (group_index + 1) * group_size, :]
            group_winner_indices = (np.argmax(group, axis=0) +
                                    group_index * group_size)
            for i in range(group_winner_indices.shape[0]):
                winner_index = group_winner_indices[i]
                l1_firing_rate[winner_index, i] = 1

        l2_input_activity = np.dot(self._weights_l2, l1_firing_rate)
        winners = np.argmax(l2_input_activity, axis=0)
        return winners
