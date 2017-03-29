"""Utility functions for loading MNIST data."""

import csv
import numpy as np
from pickle import load

from .data import Data

DEFAULT_DATA_FILE_CSV = 'digits/train.csv'
DEFAULT_LABELS_FILE_CSV = 'digits/trainlabels.csv'

DEFAULT_DATA_FILE_PICKLE = 'digits/train.pickle'
DEFAULT_LABELS_FILE_PICKLE = 'digits/trainlabels.pickle'


def load_data_csv(data_file=DEFAULT_DATA_FILE_CSV,
                  labels_file=DEFAULT_LABELS_FILE_CSV):
    """Load MNIST data from a CSV file (slow)."""
    raw_data = np.genfromtxt(data_file, delimiter=",").T
    labels = np.genfromtxt(labels_file, delimiter=",")
    return Data(raw_data=raw_data, labels=labels)


def load_data_pickle(data_file=DEFAULT_DATA_FILE_PICKLE,
                     labels_file=DEFAULT_LABELS_FILE_PICKLE):
    """Load MNIST data from a Pickle (faster)."""
    with open(data_file, 'rb') as f:
        raw_data = load(f)
    with open(labels_file, 'rb') as f:
        labels = load(f)
    return Data(raw_data=raw_data, labels=labels)
