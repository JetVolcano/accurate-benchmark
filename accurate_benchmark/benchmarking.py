import logging
from collections import deque
from collections.abc import Callable
from decimal import Decimal
from functools import partial, update_wrapper
from itertools import repeat
from time import perf_counter_ns
from typing import Final, ParamSpec, TypeVar

import numpy as np
from babel.core import default_locale
from babel.numbers import format_decimal
from scipy.stats import trim_mean

from accurate_benchmark.parameters import SingleParam

P = ParamSpec("P")
R = TypeVar("R")


def _run_func(
    func: Callable[P, R],
    acc: int,
    logger: logging.Logger,
    *args: P.args,
    **kwargs: P.kwargs,
) -> np.ndarray[int]:
    results: deque[float] = deque(maxlen=acc)
    i: int = 0
    max_log_len: int = 5
    if acc <= max_log_len:
        max_log_len = 1
    for _ in repeat(None, acc):
        i += 1
        if (i % (acc / max_log_len) == 0) or max_log_len == 1:
            logger.info("Run (%s/%s)", i, acc)
        if args == ():
            start_time: float = perf_counter_ns()
            func(**kwargs)
            end_time: float = perf_counter_ns()
        elif isinstance(args[0], SingleParam):
            start_time: float = perf_counter_ns()
            func(args[0].value, **kwargs)
            end_time: float = perf_counter_ns()
        else:
            start_time: float = perf_counter_ns()
            func(*args, **kwargs)
            end_time: float = perf_counter_ns()
        results.append(end_time - start_time)
    return np.array(results)


class Benchmark:
    """
    A class to benchmark a function by running it multiple times and printing the average time taken.
    """

    UNITS: Final[tuple[str]] = ("ns", "us", "ms", "s")

    def __init__(
        self,
        func: Callable[P, R],
        precision: int = 15,
        unit: str = "s",
        method: str = "trim_mean",
    ) -> None:
        """
        :param func: The function to benchmark.
        :param precision: The number of times to run the function to get an average time.
        :param unit: The unit of the benchmark.
        :param method: The method to use to calculate the benchmarkl
        :type func: Callable[P, R]
        :type precision: int
        :type unit: str
        :type method: str
        """
        self.__methods: dict[str, Callable] = {
            "trim_mean": partial(trim_mean, proportiontocut=0.05),
            "mean": np.mean,
            "median": np.median,
            "min": np.min,
            "max": np.max,
            "std": np.std,
            "var": np.var,
            "sum": np.sum,
            "prod": np.prod,
            "argmin": np.argmin,
            "argmax": np.argmax,
        }
        if method not in self.__methods:
            raise ValueError(f"Invalid method: {method}")
        if not isinstance(func, Callable):
            raise TypeError("func must be of type Callable")
        if not isinstance(precision, int):
            raise TypeError("precision must be of type int")
        if precision < 1:
            raise ValueError("precision must be greater than or equal to 1")
        if unit not in Benchmark.UNITS:
            raise ValueError(
                f"Invalid Unit: {unit}, Please use 'ns', 'us', 'ms', or 's'."
            )
        update_wrapper(self, func)
        self.__method: str = method
        self.__func: Callable = func
        self.__unit: str = unit
        self.__precision: int = precision
        self.__results: np.ndarray
        self.__result: Decimal
        self.__logger: logging.Logger = logging.getLogger(
            f"benchmark.{self.__name__}.{id(self)}"
        )
        self.__logger.propagate = False
        self.__logger.setLevel(logging.INFO)
        if self.__logger.hasHandlers():
            self.__logger.handlers.clear()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%a, %B %d, %Y at %I:%M:%S %p",
        )
        handler: logging.StreamHandler[logging.TextIO] = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)


    def __repr__(self) -> str:
        """
        Returns the representation of the Benchmark instance

        :returntype str:
        """
        return f"Benchmark(func={self.__func.__name__}, precision={self.__precision}, unit={self.__unit}, method={self.__method!r})"

    @property
    def precision(self) -> int:
        """
        Returns the precision of the benchmark.

        :returntype int:
        """
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
        """
        Returns the unit of the benchmark.

        :returntype str:
        """
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
        """
        Returns the method of calculating the benchmark.

        :returntype str:
        """
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        if value not in self.__methods:
            raise ValueError(f"Invalid method: {value}")
        self.__method = value

    @property
    def func(self) -> Callable[P, R]:
        """
        Returns the function being benchmarked.

        :returntype Callable[P, R]:
        """
        return self.__func

    @property
    def result(self) -> Decimal | None:
        """
        Returns the result of the benchmark if the benchmark hasn't been run yet it will return None.

        :returntype Decimal | None:
        """
        return self.__result

    def __format_function(self, *args: P.args, **kwargs: P.kwargs) -> str:
        arg_strs: deque[str] = deque()
        for arg in args:
            if isinstance(arg, SingleParam):
                arg_strs.append(repr(arg.value))
            else:
                arg_strs.append(repr(arg))

        kwarg_strs: deque[str] = deque([f"{k}={v!r}" for k, v in kwargs.items()])
        all_args: str = ", ".join(arg_strs + kwarg_strs)
        if self.__func.__module__ not in ["builtins", "__main__"]:
            return f"{self.__func.__module__}.{self.__func.__name__}({all_args})"
        return f"{self.__func.__name__}({all_args})"

    def benchmark(self, *args: P.args, **kwargs: P.kwargs) -> float:
        no_args: bool = len(args) == 0
        if not no_args:
            single_arg: bool = isinstance(args[0], SingleParam)
        if no_args:
            self.__logger.info("Benchmarking %s", self.__format_function(**kwargs))
            self.__results = _run_func(
                self.__func, self.__precision, self.__logger, **kwargs
            )
        elif single_arg:
            self.__logger.info(
                "Benchmarking %s", self.__format_function(args[0].value, **kwargs)
            )
            self.__results = _run_func(
                self.__func, self.__precision, self.__logger, args[0], **kwargs
            )
        elif not single_arg:
            self.__logger.info(
                "Benchmarking %s", self.__format_function(*args, **kwargs)
            )
            self.__results = _run_func(
                self.__func, self.__precision, self.__logger, *args, **kwargs
            )
        current_locale: str = default_locale("LC_NUMERIC") or "en_US"
        self.__result = int(self.__methods[self.__method](self.__results))
        unit: Decimal = Decimal(self.__result) / (
            1000 ** Benchmark.UNITS.index(self.__unit)
        )
        formatted: str = format_decimal(
            unit, locale=current_locale, decimal_quantization=False
        )
        units: dict[str, str] = {
            "ns": "nanoseconds",
            "us": "microseconds",
            "ms": "milliseconds",
            "s": "seconds",
        }
        expanded_unit: str = units[self.__unit]
        if no_args:
            self.__logger.info(
                f"\n{self.__format_function(**kwargs)} took {formatted} {expanded_unit}"
            )
        elif single_arg:
            self.__logger.info(
                f"\n{self.__format_function(args[0].value, **kwargs)} took {formatted} {expanded_unit}"
            )
        elif not single_arg:
            self.__logger.info(
                f"\n{self.__format_function(*args, **kwargs)} took {formatted} {expanded_unit}"
            )
        return self.__result

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__func(*args, **kwargs)

    def compare(
        self,
        func2: Callable[P, R],
        args1: tuple | SingleParam | None = None,
        args2: tuple | SingleParam | None = None,
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
        benchmark = Benchmark(func2, self.__precision, self.__unit)
        if isinstance(args1, SingleParam):
            time1: float = self.benchmark(args1, **kwargs1)
        else:
            time1: float = self.benchmark(*args1, **kwargs1)
        if isinstance(args2, SingleParam):
            time2: float = benchmark.benchmark(args2.value, **kwargs2)
        else:
            time2: float = benchmark.benchmark(*args2, **kwargs2)
        self.__precision = precision
        if not isinstance(args1, SingleParam) and not isinstance(args2, SingleParam):
            self.__logger.info(
                f"\n{self.__format_function(*args1, **kwargs1)} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {benchmark.__format_function(*args2, **kwargs2)}\n"
            )
        if isinstance(args1, SingleParam) and not isinstance(args2, SingleParam):
            self.__logger.info(
                f"\n{self.__format_function(args1.value, **kwargs1)} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {benchmark.__format_function(*args2, **kwargs2)}\n"
            )
        if not isinstance(args1, SingleParam) and isinstance(args2, SingleParam):
            self.__logger.info(
                f"\n{self.__format_function(*args1, **kwargs1)} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {benchmark.__format_function(args2.value, **kwargs2)}\n"
            )
        if isinstance(args1, SingleParam) and isinstance(args2, SingleParam):
            self.__logger.info(
                f"\n{self.__format_function(args1.value, **kwargs1)} is {time2 / time1 if time1 < time2 else time1 / time2:4f} times {'faster' if time1 < time2 else 'slower' if time2 < time1 else 'the same'} than {benchmark.__format_function(args2.value, **kwargs2)}\n"
            )
