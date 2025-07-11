from accurate_benchmark.parameters import SingleParam
from collections import deque
from collections.abc import Callable
from functools import lru_cache
from time import perf_counter
from typing import ParamSpec, TypeVar
from itertools import repeat
from scipy.stats import trim_mean


P = ParamSpec("P")
R = TypeVar("R")


@lru_cache(4096)
def _run_func(func: Callable[P, R], acc, *args: P.args, **kwargs: P.kwargs) -> float:
    results: deque[float] = deque(maxlen=acc)
    for _ in repeat(None, acc):
        if isinstance(args, SingleParam):
            start_time: float = perf_counter()
            func(args[0].value, **kwargs)
            end_time: float = perf_counter()
            results.append(end_time - start_time)
        else:
            start_time: float = perf_counter()
            func(*args, **kwargs)
            end_time: float = perf_counter()
            results.append(end_time - start_time)
    return trim_mean(results, 0.05)


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
        self.__func: Callable = func
        self.__precision: int = precision
        self.__result: float = ...
        self.__doc__: str | None = self.__func.__doc__
        self.__name__: str = self.__func.__name__

    @property
    def func(self) -> Callable[P, R]:
        """
        Returns the function being benchmarked.

        :returntype Callable[P, R]:
        """
        return self.__func

    def __format_function(self, *args: P.args, **kwargs: P.kwargs) -> str:
        arg_strs: deque[str] = deque()
        for arg in args:
            if isinstance(arg, SingleParam):
                arg_strs.append(repr(arg.value))
            else:
                arg_strs.append(repr(arg))
        kwarg_strs: deque[str] = deque([f"{k}={repr(v)}" for k, v in kwargs.items()])
        all_args: str = ", ".join(arg_strs + kwarg_strs)
        return f"{self.__func.__name__}({all_args})"

    def benchmark(self, *args: P.args, **kwargs: P.kwargs) -> float:
        if len(args) == 0:
            self.__result = _run_func(self.__func, self.__precision, **kwargs)
        elif isinstance(args[0], SingleParam):
            self.__result = _run_func(
                self.__func, self.__precision, args[0].value, **kwargs
            )
        else:
            self.__result = _run_func(self.__func, self.__precision, *args, **kwargs)
        if len(args) == 0:
            print(
                f"{self.__format_function(**kwargs)} took {self.__result:.18f} seconds"
            )
        elif not isinstance(args[0], SingleParam):
            print(
                f"{self.__format_function(*args, **kwargs)} took {self.__result:.18f} seconds"
            )
        else:
            print(
                f"{self.__format_function(args[0].value, **kwargs)} took {self.__result:.18f} seconds"
            )
        return self.__result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__func(*args, **kwargs)

    def compare(
        self,
        func2: Callable[P, R],
        args1: tuple | None = None,
        args2: tuple | None = None,
        accuracy: int = ...,
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
        if args1 is None:
            args1 = ()
        if args2 is None:
            args2 = ()
        if kwargs1 == ...:
            kwargs1 = {}
        if kwargs2 == ...:
            kwargs2 = {}
        precision: int = self.__precision
        if accuracy is not ...:
            self.__precision = accuracy
        benchmark = Benchmark(func2, self.__precision)
        if isinstance(args1, SingleParam):
            time1: float = self.benchmark(args1.value, **kwargs1)
        else:
            time1: float = self.benchmark(*args1, **kwargs1)
        if isinstance(args2, SingleParam):
            time2: float = benchmark.benchmark(args2.value, **kwargs2)
        else:
            time2: float = benchmark.benchmark(*args2, **kwargs2)
        self.__precision = precision
        print(
            f"{self.__func.__name__} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {func2.__name__}"
        )
