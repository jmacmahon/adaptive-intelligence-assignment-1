from random import randrange
import matplotlib.pyplot as plt

from assignment.load import load_data_pickle
from assignment.dimensionality import PCAReducer
from assignment.neural import SingleLayerCompetitiveNetwork
from assignment.display import show_image, create_weights_plot

def random_iter(indexable, n):
    for _ in range(n):
        index = randrange(len(indexable))
        yield indexable[index]

def every(iter, n):
    try:
        while True:
            yield next(iter)
            for _ in zip(range(n - 1), iter):
                pass
    except StopIteration:
        raise

def consume(iter):
    for _ in iter:
        pass

data = load_data_pickle()
# data.reducer = PCAReducer(5)

nn = SingleLayerCompetitiveNetwork(28*28, 15)

training_gen = nn.train_many(random_iter(data, 50000))


def show_weights():
    show_units = create_weights_plot(5, 3)
    show_units(nn._weights)

def train_and_show():
    show_units = create_weights_plot(5, 3)
    for _, _, weights in every(training_gen, 1000):
        show_units(weights)
