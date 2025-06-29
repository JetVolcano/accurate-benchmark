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

    async def benchmark(self, *args: P.args, **kwargs: P.kwargs) -> float:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        results: deque[float] = deque(maxlen=self.__precision)
        parameters: tuple[object] = args + tuple(kwargs.items())
        with ProcessPoolExecutor() as executor:
            tasks: deque = deque(
                [
                    loop.run_in_executor(
                        executor, partial(_run_func, self.__func, *args, **kwargs)
                    )
                    for _ in repeat(None, self.__precision)
                ]
            )
            for task in tasks:
                duration: float = await task
                results.append(duration)
        self.__result = trim_mean(results, 0.05)
        print(f"{self.__func.__name__}{parameters} took {self.__result:.18f} seconds")
        return self.__result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__func(*args, **kwargs)

    async def compare(
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
        if not isinstance(args1, Iterable):
            time1 = await self.benchmark(*[args1], **kwargs1)
        else:
            time1 = await self.benchmark(*args1, **kwargs1)
        if not isinstance(args2, Iterable):
            time2 = await benchmark.benchmark(*[args2], **kwargs2)
        else:
            time2 = await benchmark.benchmark(*args2, **kwargs2)
        self.__precision = precision
        print(
            f"{self.__func.__name__} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {func2.__name__}"
        )
