from random import randrange
from time import time

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
