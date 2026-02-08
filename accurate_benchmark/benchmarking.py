from collections import deque
from collections.abc import Awaitable, Callable
from decimal import Decimal
from functools import partial, update_wrapper
from itertools import repeat
from time import perf_counter_ns
from typing import Any, Final, Generic, ParamSpec, TypeVar

import numpy as np
from babel.core import default_locale
from babel.numbers import format_decimal
from rich.console import Console
from scipy.stats import trim_mean

from accurate_benchmark._console import _create_console
from accurate_benchmark.parameters import SingleParam

P = ParamSpec("P")
R = TypeVar("R")


def _format_function(func: Callable[P, R], *args: Any, **kwargs: Any) -> str:
    arg_strs: deque[str] = deque([repr(arg) for arg in args])
    kwarg_strs: deque[str] = deque([f"{k}={v!r}" for k, v in kwargs.items()])
    all_args: str = ", ".join([*arg_strs, *kwarg_strs])
    if func.__module__ not in ["builtins", "__main__"]:
        return f"{func.__module__}.{func.__name__}({all_args})"
    return f"{func.__name__}({all_args})"


def _run_func(
    func: Callable[..., R],
    acc: int,
    console: Console,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    results: deque[int | float] = deque(maxlen=acc)
    i: int = 0
    max_log_len: int = 5
    if acc <= max_log_len:
        max_log_len = 1
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    for _ in repeat(None, acc):
        i += 1
        start_time: float = perf_counter_ns()
        func(*modded_args, **kwargs)
        end_time: float = perf_counter_ns()
        if (i % (acc // max_log_len) == 0) or max_log_len == 1:
            console.log(f"Run ({i}/{acc}) Completed")
        results.append(end_time - start_time)
    return np.array(results)


async def _async_run_func(
    func: Callable[..., Awaitable[R]],
    acc: int,
    console: Console,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    results: deque[int] = deque(maxlen=acc)
    i: int = 0
    max_log_len: int = 5
    if acc <= max_log_len:
        max_log_len = 1
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    for _ in repeat(None, acc):
        i += 1
        start_time: float = perf_counter_ns()
        await func(*modded_args, **kwargs)
        end_time: float = perf_counter_ns()
        if (i % (acc // max_log_len) == 0) or max_log_len == 1:
            console.log(f"Run ({i}/{acc}) Completed")
        results.append(end_time - start_time)
    return np.array(results)


class Benchmark(Generic[P, R]):
    """
    A class for benchmarking synchronous functions by running them mulitple times
    and logging the result of the values using the method inputed
    """

    UNITS: Final[tuple[str, str, str, str]] = ("ns", "us", "ms", "s")

    def __init__(
        self,
        func: Callable[P, R],
        precision: int = 15,
        unit: str = "s",
        method: str = "trim_mean",
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[P, R]
            The  function to benchmark.
        precision : int, optional, default=15
            The number of times to run the function to get an average time.
        unit : str, optional, default="s"
            The unit of time that wil be outputed.
        method : str, optional, default="trim_mean"
            The method that will be used to get the time. (default is the most accurate)

        Raises
        ------
        ValueError
            Method is not supported, (supported methods: trim_mean, mean, median)
        TypeError
            func must be of type Callable.
        TypeError
            precision must be of type int.
        ValueError
            precision must be greater than or equal to 1.
        ValueError
            Unit does not exist, (supported units: ns, us, ms, s)
        """
        self.__methods: dict[str, Callable] = {
            "trim_mean": partial(trim_mean, proportiontocut=0.05),
            "mean": np.mean,
            "median": np.median,
        }
        if method not in self.__methods:
            raise ValueError(
                f"Method is not supported: {method}, (supported methods: {', '.join(self.__methods.keys())})"
            )
        if not isinstance(func, Callable):
            raise TypeError("func must be of type Callable.")
        if not isinstance(precision, int):
            raise TypeError("precision must be of type int.")
        if precision < 1:
            raise ValueError("precision must be greater than or equal to 1.")
        if unit not in Benchmark.UNITS:
            raise ValueError(
                f"Unit does not exist: {unit}, (supported units: {', '.join(Benchmark.UNITS)})"
            )
        update_wrapper(self, func)
        self.__method: str = method
        self.__func: Callable[P, R] = func
        self.__unit: str = unit
        self.__precision: int = precision
        self.__results: np.ndarray
        self.__result: int | float
        self.__console: Console = _create_console()

    def __repr__(self) -> str:
        return f"Benchmark(func={self.__func.__name__}, precision={self.__precision}, unit={self.__unit!r}, method={self.__method!r})"

    @property
    def precision(self) -> int:
        return self.__precision

    @precision.setter
    def precision(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("precision must be of type int")
        if value < 1:
            raise ValueError("precision must be greater than or equal to 1")
        self.__precision = value

    @property
    def unit(self) -> str:
        return self.__unit

    @unit.setter
    def unit(self, value: str) -> None:
        if value not in Benchmark.UNITS:
            raise ValueError(
                f"Invalid Unit: {value}, Please use 'ns', 'us', 'ms', or 's'."
            )
        self.__unit = value

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        if value not in self.__methods:
            raise ValueError(f"Invalid method: {value}")
        self.__method = value

    @property
    def func(self) -> Callable[P, R]:
        return self.__func

    @property
    def result(self) -> int | float | None:
        return self.__result

    def benchmark(self, *args: Any, **kwargs: Any) -> int | float:
        self.__console.rule(
            f"Benchmarking {_format_function(self.__func, *args, **kwargs)}",
        )
        self.__results = _run_func(
            self.__func, self.__precision, self.__console, *args, **kwargs
        )
        current_locale: str = default_locale("LC_NUMERIC") or "en_US"
        self.__result = self.__methods[self.__method](self.__results)
        unit: Decimal = Decimal(self.__result) / Decimal(
            1000 ** Benchmark.UNITS.index(self.__unit)
        )
        formatted: str = format_decimal(unit, locale=current_locale)
        units: dict[str, str] = {
            "ns": "nanoseconds",
            "us": "microseconds",
            "ms": "milliseconds",
            "s": "seconds",
        }
        expanded_unit: str = units[self.__unit]
        if unit == 1:
            expanded_unit = expanded_unit[:-1]
        self.__console.log(
            f"{_format_function(self.__func, *args, **kwargs)} took {formatted} {expanded_unit}"
        )
        return self.__result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__func(*args, **kwargs)

    def compare(
        self,
        func2: Callable[P, R],
        args1: tuple | SingleParam | None = None,
        args2: tuple | SingleParam | None = None,
        kwargs1: dict | None = None,
        kwargs2: dict | None = None,
        accuracy: int | None = None,
    ) -> None:
        """Compare the exectution time of two functions.

        Parameters
        ----------
        func2 : Callable[P, R]
            The second function to benchmark.
        args1 : tuple | SingleParam | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple | SingleParam | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict | None, optional, default=None
            The keyword arguments for the second function
        accuracy : int | None, optional, default=None
            How many times to run each function.
        """
        if args1 is None:
            args1 = ()
        if args2 is None:
            args2 = ()
        if kwargs1 is None:
            kwargs1 = {}
        if kwargs2 is None:
            kwargs2 = {}
        precision: int = self.__precision
        if accuracy is not None:
            self.__precision = accuracy
        benchmark: Benchmark[P, R] = Benchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        time1: float = self.benchmark(*args1, **kwargs1)
        time2: float = benchmark.benchmark(*args2, **kwargs2)
        self.__precision = precision
        formatted_self: str = _format_function(self.__func, *args1, **kwargs1)
        formatted_other: str = _format_function(func2, *args2, **kwargs2)
        times_slower_faster: str = f"{(time2 / (time1 + 1) if time1 < 1 else (time2 / time1 if time1 < time2 else time2)) if time1 < time2 else (time1 / (time2 + 1) if time2 < 1 else time1 / time2):4f}"
        self.__console.rule("Results")
        self.__console.log(
            f"{formatted_self} is {f'{times_slower_faster} times faster than {formatted_other}' if time1 < time2 else f'{times_slower_faster} times slower than {formatted_other}' if time2 < time1 else f'the same as {formatted_other}'}"
        )


class AsyncBenchmark(Generic[P, R]):
    """
    A class for benchmarking asynchronous functions by running them mulitple times
    and logging the result of the values using the method inputed
    """

    def __init__(
        self,
        func: Callable[P, Awaitable[R]],
        precision: int = 15,
        unit: str = "s",
        method: str = "trim_mean",
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[P, R]
            The  function to benchmark.
        precision : int, optional, default=15
            The number of times to run the function to get an average time.
        unit : str, optional, default="s"
            The unit of time that wil be outputed.
        method : str, optional, default="trim_mean"
            The method that will be used to get the time. (default is the most accurate)

        Raises
        ------
        ValueError
            Method is not supported, (supported methods: trim_mean, mean, median)
        TypeError
            func must be of type Callable.
        TypeError
            precision must be of type int.
        ValueError
            precision must be greater than or equal to 1.
        ValueError
            Unit does not exist, (supported units: ns, us, ms, s)
        """
        self.__methods: dict[str, Callable] = {
            "trim_mean": partial(trim_mean, proportiontocut=0.05),
            "mean": np.mean,
            "median": np.median,
        }
        if method not in self.__methods:
            raise ValueError(
                f"Method is not supported: {method}, (supported methods: {', '.join(self.__methods.keys())})"
            )
        if not isinstance(func, Callable):
            raise TypeError("func must be of type Callable.")
        if not isinstance(precision, int):
            raise TypeError("precision must be of type int.")
        if precision < 1:
            raise ValueError("precision must be greater than or equal to 1.")
        if unit not in Benchmark.UNITS:
            raise ValueError(
                f"Unit does not exist: {unit}, (supported units: {', '.join(Benchmark.UNITS)})"
            )
        update_wrapper(self, func)
        self.__method: str = method
        self.__func: Callable[P, Awaitable[R]] = func
        self.__unit: str = unit
        self.__precision: int = precision
        self.__results: np.ndarray
        self.__result: int | float
        self.__console: Console = _create_console()

    def __repr__(self) -> str:
        return f"AsyncBenchmark(func={self.__func.__name__}, precision={self.__precision}, unit={self.__unit!r}, method={self.__method!r})"

    @property
    def precision(self) -> int:
        return self.__precision

    @precision.setter
    def precision(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError("precision must be of type int")
        if value < 1:
            raise ValueError("precision must be greater than or equal to 1")
        self.__precision = value

    @property
    def unit(self) -> str:
        return self.__unit

    @unit.setter
    def unit(self, value: str) -> None:
        if value not in Benchmark.UNITS:
            raise ValueError(
                f"Invalid Unit: {value}, Please use 'ns', 'us', 'ms', or 's'."
            )
        self.__unit = value

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        if value not in self.__methods:
            raise ValueError(f"Invalid method: {value}")
        self.__method = value

    @property
    def func(self) -> Callable[P, Awaitable[R]]:
        return self.__func

    @property
    def result(self) -> int | float | None:
        return self.__result

    async def benchmark(self, *args: Any, **kwargs: Any) -> int | float:
        self.__console.rule(
            f"Benchmarking {_format_function(self.__func, *args, **kwargs)}",
        )
        self.__results = await _async_run_func(
            self.__func, self.__precision, self.__console, *args, **kwargs
        )
        current_locale: str = default_locale("LC_NUMERIC") or "en_US"
        self.__result = self.__methods[self.__method](self.__results)
        unit: Decimal = Decimal(self.__result) / Decimal(
            1000 ** Benchmark.UNITS.index(self.__unit)
        )
        formatted: str = format_decimal(unit, locale=current_locale)
        units: dict[str, str] = {
            "ns": "nanoseconds",
            "us": "microseconds",
            "ms": "milliseconds",
            "s": "seconds",
        }
        expanded_unit: str = units[self.__unit]
        if unit == 1:
            expanded_unit = expanded_unit[:-1]
        self.__console.log(
            f"{_format_function(self.__func, *args, **kwargs)} took {formatted} {expanded_unit}"
        )
        return self.__result

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self.__func(*args, **kwargs)

    async def compare(
        self,
        func2: Callable[P, Awaitable[R]],
        args1: tuple | SingleParam | None = None,
        args2: tuple | SingleParam | None = None,
        kwargs1: dict | None = None,
        kwargs2: dict | None = None,
        accuracy: int | None = None,
    ) -> None:
        """Compare the exectution time of two functions.

        Parameters
        ----------
        func2 : Callable[P, Awaitable[R]]
            The second asynchronous function to benchmark.
        args1 : tuple | SingleParam | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple | SingleParam | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict | None, optional, default=None
            The keyword arguments for the second function
        accuracy : int | None, optional, default=None
            How many times to run each function.
        """
        if args1 is None:
            args1 = ()
        if args2 is None:
            args2 = ()
        if kwargs1 is None:
            kwargs1 = {}
        if kwargs2 is None:
            kwargs2 = {}
        precision: int = self.__precision
        if accuracy is not None:
            self.__precision = accuracy
        benchmark: AsyncBenchmark[P, R] = AsyncBenchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        time1: float = await self.benchmark(*args1, **kwargs1)
        time2: float = await benchmark.benchmark(*args2, **kwargs2)
        self.__precision = precision
        formatted_self: str = _format_function(self.__func, *args1, **kwargs1)
        formatted_other: str = _format_function(func2, *args2, **kwargs2)
        times_slower_faster: str = f"{(time2 / (time1 + 1) if time1 < 1 else (time2 / time1 if time1 < time2 else time2)) if time1 < time2 else (time1 / (time2 + 1) if time2 < 1 else time1 / time2):4f}"
        self.__console.rule("Results")
        self.__console.log(
            f"{formatted_self} is {f'{times_slower_faster} times faster than {formatted_other}' if time1 < time2 else f'{times_slower_faster} times slower than {formatted_other}' if time2 < time1 else f'the same as {formatted_other}'}"
        )


async def compare(
    bench1: Benchmark | AsyncBenchmark,
    bench2: Benchmark | AsyncBenchmark,
    args1: tuple | SingleParam | None = None,
    args2: tuple | SingleParam | None = None,
    kwargs1: dict | None = None,
    kwargs2: dict | None = None,
) -> None:
    """Compare the exectution time of two benchmarks.
    Use this if you want to compare an async function with a synchronous function

    Parameters
    ----------
    bench1 : Benchmark | AsyncBenchmark
        The first benchmark to compare.
    bench2 : Benchmark | AsyncBenchmark
        The second benchmark to compare.
    args1 : tuple | SingleParam | None, optional, default=None
        The posistional arguments for the first function.
    args2 : tuple | SingleParam | None, optional, default=None
        The posistional arguments for the second function.
    kwargs1: dict | None, optional, default=None
        The keyword arguments for the first function
    kwargs2: dict | None, optional, default=None
        The keyword arguments for the second function
    """
    if args1 is None:
        args1 = ()
    if args2 is None:
        args2 = ()
    if kwargs1 is None:
        kwargs1 = {}
    if kwargs2 is None:
        kwargs2 = {}
    time1: float
    time2: float
    console: Console = _create_console()
    if isinstance(bench1, Benchmark):
        time1 = bench1.benchmark(*args1, **kwargs1)
    elif isinstance(bench1, AsyncBenchmark):
        time1 = await bench1.benchmark(*args1, **kwargs1)
    if isinstance(bench2, Benchmark):
        time2 = bench2.benchmark(*args2, **kwargs2)
    elif isinstance(bench2, AsyncBenchmark):
        time2 = await bench2.benchmark(*args2, **kwargs2)
    formatted_self: str = _format_function(bench1.func, *args1, **kwargs1)
    formatted_other: str = _format_function(bench2.func, *args2, **kwargs2)
    times_slower_faster: str = f"{(time2 / (time1 + 1) if time1 < 1 else (time2 / time1 if time1 < time2 else time2)) if time1 < time2 else (time1 / (time2 + 1) if time2 < 1 else time1 / time2):4f}"
    console.rule("Results")
    console.log(
        f"{formatted_self} is {f'{times_slower_faster} times faster than {formatted_other}' if time1 < time2 else f'{times_slower_faster} times slower than {formatted_other}' if time2 < time1 else f'the same as {formatted_other}'}"
    )
