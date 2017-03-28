from random import randrange
import matplotlib.pyplot as plt
from time import time
import numpy as np
from functools import partial

from assignment.load import load_data_pickle
from assignment.dimensionality import PCAReducer, DropFirstNSelector
from assignment.neural import (SingleLayerCompetitiveNetwork,
    TwoLayerCompetitiveNetwork)
from assignment.display import (show_image, create_weights_plot,
    show_3d_classes, get_3d_figures, show_3d_tunings)
from assignment.data import Data
from assignment.evaluation import evaluate, fuzz_evaluate
from assignment.util import consume, random_iter, every, count_every

data = load_data_pickle()
data.normalise()
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
    for _, weights, _ in every(training_gen, 1000):
        show_units(data.reconstruct(weights))

def train_and_show_2l(nn, data, n=50000):
    training_gen = nn.train_many(random_iter(data._raw_data, n))
    show_units = create_weights_plot(50)
    for _, weights, _ in every(training_gen, 1000):
        show_units(data.reconstruct(weights))
