from accurate_benchmark.parameters import SingleParam
from collections import deque
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import perf_counter
from typing import ParamSpec, TypeVar
from itertools import repeat
from scipy.stats import trim_mean
import asyncio


P = ParamSpec("P")
R = TypeVar("R")


def _run_func(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    if isinstance(*args, SingleParam):
        start_time: float = perf_counter()
        func(args[0],  **kwargs)
        end_time: float = perf_counter()
    else:
        start_time: float = perf_counter()
        func(*args, **kwargs)
        end_time: float = perf_counter()
    return end_time - start_time


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
        results: deque[float] = deque(maxlen=self.__precision)
        for _ in repeat(None, self.__precision):
            if isinstance(args[0], SingleParam):
                start_time: float = perf_counter()
                self.__func(args[0].value, **kwargs)
                end_time: float = perf_counter()
            else:
                start_time: float = perf_counter()
                self.__func(*args, **kwargs)
                end_time: float = perf_counter()
            results.append(end_time - start_time)
        self.__result = trim_mean(results, 0.05)
        if not isinstance(args[0], SingleParam):
            print(
                f"{self.__format_function(*args, **kwargs)} took {self.__result:.18f} seconds"
            )
        else:
            print(
                f"{self.__format_function(args[0].value, **kwargs)} took {self.__result:.18f} seconds"
            )
        return self.__result

    async def async_benchmark(self, *args: P.args, **kwargs: P.kwargs) -> float:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        results: deque[float] = deque(maxlen=self.__precision)
        with ProcessPoolExecutor() as executor:
            tasks: deque = deque(
                [
                    loop.run_in_executor(
                        executor, partial(_run_func, self.__func, *args, **kwargs)
                    )
                    if not isinstance(args[0], SingleParam)
                    else loop.run_in_executor(
                        _run_func, self.__func, args[0].value, **kwargs
                    )
                    for _ in repeat(None, self.__precision)
                ]
            )
            for task in tasks:
                duration: float = await task
                results.append(duration)
        self.__result = trim_mean(results, 0.05)
        if not isinstance(*args, SingleParam):
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

    async def async_compare(
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
            time1: float = await self.async_benchmark(args1.value, **kwargs1)
        else:
            time1: float = await self.async_benchmark(*args1, **kwargs1)
        if isinstance(args2, SingleParam):
            time2: float = await benchmark.async_benchmark(args2.value, **kwargs2)
        else:
            time2: float = await benchmark.async_benchmark(*args2, **kwargs2)
        self.__precision = precision
        print(
            f"{self.__func.__name__} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {func2.__name__}"
        )


def add(iterable: Iterable[float]) -> float:
    return sum(iterable)

def sub(a, b) -> float:
    return a - b

if __name__ == "__main__":
    bench1: Benchmark = Benchmark(add)
    bench2: Benchmark = Benchmark(sub)
    bench1.benchmark(SingleParam([1, 2, 3]))
    bench2.benchmark(5, 1)