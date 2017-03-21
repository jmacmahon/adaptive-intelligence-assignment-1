from random import randrange
import matplotlib.pyplot as plt
from time import time

from assignment.load import load_data_pickle
from assignment.dimensionality import PCAReducer, DropFirstNSelector
from assignment.neural import (SingleLayerCompetitiveNetwork,
    TwoLayerCompetitiveNetwork)
from assignment.display import (show_image, create_weights_plot,
    show_3d_classes, get_3d_figures)
from assignment.data import Data

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
data.reducer = PCAReducer(100)
reduced = data.reduce()

def create_and_train(class_, data, n, outputs=15, **kwargs):
    nn = class_(inputs=data.dimensions, outputs=outputs, **kwargs)
    consume(nn.train_many(random_iter(data, n)))
    return nn

def show_weights(nn, w=5, h=3):
    show_units = create_weights_plot(w, h)
    show_units(nn._weights)

def train_and_show(nn, data, n=50000):
    training_gen = nn.train_many(random_iter(data, n))
    show_units = create_weights_plot(15)
    for _, weights in every(training_gen, 1000):
        show_units(data.reconstruct(weights))

def train_and_show_2l(nn, data, n=50000):
    training_gen = nn.train_many(random_iter(data._raw_data, n))
    show_units = create_weights_plot(50)
    for _, weights, _ in every(training_gen, 1000):
        show_units(data.reconstruct(weights))

def count_every(iterable, n=10000, total=None):
    ii = 0
    start_t = t = time()
    for elem in iterable:
        if ii % n == 0:
            new_t = time()
            dt_total = float(new_t - start_t)
            ditems_dt_avg = float(ii)/dt_total
            dt_block = float(new_t - t)
            ditems_dt_block = float(n)/dt_block
            t = new_t
            print('Done %d in %.1fs (average: %.1f/s, block: %.1f/s)' %
                  (ii, dt_total, ditems_dt_avg, ditems_dt_block))

            if total is not None:
                percentage = 100 * float(ii)/total
                try:
                    eta = (total - ii)/ditems_dt_avg
                except ZeroDivisionError:
                    eta = 0
                print('Completion: %.1f%%; ETA: %.0fs' % (percentage, eta))
        ii += 1
        yield elem
