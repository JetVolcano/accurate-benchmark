from collections import deque
from collections.abc import Callable
from functools import wraps, lru_cache
from time import perf_counter
from itertools import repeat
from typing import Any


@lru_cache(maxsize=4096)
def benchmark(func: Callable) -> Callable:
    """
    Decorator to benchmark a function
    ---
    when there is a TypeError when decorating your function or testing a function that that isn't in your code please use the base function instead of the decorator like this:
    benchmark(func)(args)\n
    :param func: "function to benchmark"\n
    :type func: "Callable"\n
    :return: "docstring of parameter func"
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> None:
        results: deque[float | int] = deque()
        parameters: tuple[object] = args + tuple(kwargs.items())
        for _ in repeat(None, 15):
            try:
                start = perf_counter()
                output = func(*args, **kwargs)
                end = perf_counter()
                results.append(end - start)
            except Exception as e:
                print(f"Exception in {func.__name__}: {e}")
                break
        average: float = sum(results) / len(results)
        print(f"Benchmarking {func.__name__}{parameters} took {average:.12f} seconds")

    return wrapper
