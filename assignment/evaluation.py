from multiprocessing import Pool
from scipy.stats.distributions import t
import numpy as np
from itertools import product
from functools import partial

from .util import consume, random_iter, count_every

DEFAULT_POOL_SIZE = 6
DEFAULT_AVERAGE_OVER = 10

def evaluate(network_factory, data, n, poolsize=DEFAULT_POOL_SIZE,
             average_over=DEFAULT_AVERAGE_OVER):
    pool = Pool(poolsize)
    networks = []
    for _ in range(average_over):
        networks.append({
            'network': network_factory(),
            'raw_data': data._raw_data,
            'labels': data._labels,
            'n': n
        })
    accuracies = np.array(pool.map(train_and_evaluate, networks))
    pool.close()
    pool.join()

    mean = np.mean(accuracies)
    stdev = np.sqrt(np.var(accuracies))/np.sqrt(10)
    t_value = t(average_over - 1).ppf(0.975)
    lower_bound = mean - t_value * stdev
    upper_bound = mean + t_value * stdev

    return {
        'mean': mean,
        'confidence_interval': (lower_bound, upper_bound)
    }


def train_and_evaluate(kwargs):
    network, raw_data, labels, n = (kwargs['network'], kwargs['raw_data'],
                                    kwargs['labels'], kwargs['n'])
    iter_ = random_iter(raw_data, n)
    consume(network.train_many(iter_))
    return network.evaluate(raw_data, labels)


def fuzz_evaluate(network_partial, params, data, n, poolsize=DEFAULT_POOL_SIZE,
                  average_over=DEFAULT_AVERAGE_OVER):
    # params is a dict of len()-able iterators to be combined

    def _inner_gen():
        for values in product(*params.values()):
            kwargs = dict(zip(params.keys(), values))
            new_partial = partial(network_partial, **kwargs)
            result = evaluate(new_partial, data, n, poolsize, average_over)
            yield (kwargs, result)

    results = []
    detailed_results = []
    total_combinations = len(list(product(*params.values())))

    for kwargs, result in count_every(_inner_gen(), n=1,
                                      total=total_combinations):
        results.append(tuple(kwargs.values()) + (result['mean'],))
        detailed_results.append((kwargs, result))
        print((kwargs, result))
    return detailed_results, np.array(results)
