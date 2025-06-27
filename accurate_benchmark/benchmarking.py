from collections import deque
from collections.abc import Callable, Iterable
from functools import wraps, lru_cache
from time import perf_counter
from typing import ParamSpec, TypeVar
from itertools import repeat
from scipy.stats import trim_mean
import psutil


P = ParamSpec("P")
R = TypeVar("R")


class Benchmark:
    """
    A class to benchmark a function by running it multiple times and printing the average time taken.
    """

    __virtual_memory = psutil.virtual_memory()

    def __init__(self, func: Callable[P, R], precision: int = 15) -> None:
        """
        :param func: The function to benchmark.
        :param precision: The number of times to run the function to get an average time.
        :type func: Callable[P, R]
        :type precision: int
        """
        self.__func: Callable = func
        self.__precision: int = precision

    def benchmark(
        self,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        def decorator() -> Callable[P, R]:
            @wraps(self.__func)
            @lru_cache(self.__virtual_memory.available // 8)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                results: deque[float] = deque(maxlen=self.__precision)
                parameters: tuple[object] = args + tuple(kwargs.items())
                for _ in repeat(None, self.__precision):
                    start_time = perf_counter()
                    self.__func(*args, **kwargs)
                    end_time = perf_counter()
                    results.append(end_time - start_time)
                result: float = trim_mean(results, 0.05)
                print(f"{self.__func.__name__}{parameters} took {result:.18f} seconds")

                return result

            return wrapper

        return decorator

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.benchmark()(self.__func)(*args, **kwargs)

    def compare(
        self,
        func2: Callable[..., R],
        args1: tuple,
        args2: tuple,
        accuracy: int = 15,
        kwargs1: dict = ...,
        kwargs2: dict = ...,
    ) -> None:
        """
        Compare the execution time of two functions with the same parameters.

        :param func2: The second function to benchmark.
        :param args1: The posistional arguments for self
        :param args2: The posistional arguments for func2
        :param kwargs1: The keyword arguments for self
        :param kwargs2: The keyword arguments for func2
        :param accuracy: How many times to run each function, a higher is more accurate than a smaller number but it takes longer
        :returntype None:
        """
        if kwargs1 == ...:
            kwargs1 = {}
        if kwargs2 == ...:
            kwargs2 = {}
        precision: int = self.__precision
        self.__precision = accuracy
        benchmark1 = self
        benchmark2 = Benchmark(func2, accuracy)
        if not isinstance(args1, Iterable):
            time1 = benchmark1(*[args1], **kwargs1)
        else:
            time1 = benchmark1(*args1, **kwargs1)
        if not isinstance(args2, Iterable):
            time2 = benchmark2(*[args2], **kwargs2)
        else:
            time2 = benchmark2(*args2, **kwargs2)
        self.__precision = precision
        print(
            f"{self.__func.__name__} is {time2 / time1:.4f} times {'faster' if time1 < time2 else 'slower'} than {func2.__name__}"
        )
