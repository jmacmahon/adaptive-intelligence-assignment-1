import csv
import numpy as np

from .data import Data

DEFAULT_DATA_FILE = 'digits/train.csv'
DEFAULT_LABELS_FILE = 'digits/trainlabels.csv'

def load_data(data_file=DEFAULT_DATA_FILE, labels_file=DEFAULT_LABELS_FILE):
    raw_data = np.genfromtxt(DEFAULT_DATA_FILE, delimiter=",").T
    labels = np.genfromtxt(DEFAULT_LABELS_FILE, delimiter=",")
    return Data(raw_data=raw_data, labels=labels)
