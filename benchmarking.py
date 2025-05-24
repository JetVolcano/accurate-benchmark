from collections import deque
from collections.abc import Callable
from functools import wraps, lru_cache
from time import perf_counter
from itertools import repeat
from typing import ParamSpec, TypeVar, Callable
import sys


P = ParamSpec("P")
R = TypeVar("R")


class Benchmark:
    """
    A class to benchmark a function by running it multiple times and printing the average time taken.
    """

    def __init__(self, func: Callable[P, R], precision: int = 15) -> None:
        """
        :param func: The function to benchmark.
        :param precision: The number of times to run the function to get an average time.
        :type func: Callable[P, R]
        :type precision: int
        """
        self.precision = precision

        @lru_cache(maxsize=sys.maxsize)
        def cbenchmark(
            precision: int = 15,
        ) -> Callable[[Callable[P, R]], Callable[P, R]]:
            def decorator(func: Callable[P, R]) -> Callable[P, R]:
                @wraps(func)
                def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                    results: deque[float] = deque(maxlen=precision)
                    parameters: tuple[object] = args + tuple(kwargs.items())
                    for _ in repeat(None, precision):
                        start_time = perf_counter()
                        func(*args, **kwargs)
                        end_time = perf_counter()
                        results.append(end_time - start_time)
                    mean: float = sum(results) / len(results)
                    print(f"{func.__name__}{parameters} took {mean:.12f} seconds")

                    return mean

                return wrapper

            return decorator

        self.__result = cbenchmark(precision)(func)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__result(*args, **kwargs)
