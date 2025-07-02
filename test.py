import numpy as np
from accurate_benchmark.parameters import SingleParam


def test(iterable):
    print(iterable)


test(SingleParam(np.random.randint(1, 100, 100)).value)