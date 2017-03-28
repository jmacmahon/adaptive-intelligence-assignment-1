from multiprocessing import Pool
from scipy.stats.distributions import t
import numpy as np
from itertools import product
from functools import partial

from .util import consume, random_iter, count_every

DEFAULT_POOL_SIZE = 5
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


def fuzz_evaluate(network_partial, params, data, n=None,
                  poolsize=DEFAULT_POOL_SIZE,
                  average_over=DEFAULT_AVERAGE_OVER):
    # params is a dict of len()-able iterators to be combined

    def _inner_gen():
        for values in product(*params.values()):
            kwargs = dict(zip(params.keys(), values))
            new_partial = partial(network_partial, **kwargs)
            if n is not None:
                this_n = n
            else:
                this_n = kwargs['n']
                del kwargs['n']
            result = evaluate(new_partial, data, this_n, poolsize, average_over)
            yield (kwargs, this_n, result)

    results = []
    detailed_results = []
    total_combinations = len(list(product(*params.values())))

    for kwargs, this_n, result in count_every(_inner_gen(), n=1,
                                      total=total_combinations):
        results.append(tuple(kwargs.values()) + (this_n, result['mean']))
        detailed_results.append((kwargs, this_n, result))
        print((kwargs, this_n, result))
    return detailed_results, np.array(results)
