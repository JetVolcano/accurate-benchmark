from collections import deque
from collections.abc import Awaitable, Callable
from decimal import Decimal
from functools import update_wrapper, wraps
from itertools import repeat
from time import perf_counter_ns
from typing import Any, Generic, Literal, ParamSpec, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
from babel.core import default_locale
from babel.numbers import format_decimal
from matplotlib.container import BarContainer
from rich.console import Console
from rich.live import Live
from rich.text import Text

from accurate_benchmark._console import _BenchmarkConsole
from accurate_benchmark._constants import Constants
from accurate_benchmark.parameters import SingleParam


P = ParamSpec("P")
R = TypeVar("R")


GlobalConsole: _BenchmarkConsole = Constants.GLOBAL_CONSOLE


def _format_function(
    func: Callable[P, R], parameters: bool = True, *args: Any, **kwargs: Any
) -> Text:
    em: list[str] = ["builtins", "__main__"]
    name: Text = Text(
        f"{f'{func.__module__}.' if func.__module__ not in em else ''}{func.__name__}",
        style="repr.call",
    )
    if not parameters:
        return name
    arg_strs: deque[str] = deque([repr(arg) for arg in args])
    kwarg_strs: deque[str] = deque([f"{k}={v!r}" for k, v in kwargs.items()])
    all_args: str = ", ".join([*arg_strs, *kwarg_strs])
    all_args_truncated: Text = Text(all_args)
    all_args_truncated.truncate(150, overflow="ellipsis")
    return Text.assemble(name, "(", all_args_truncated, ")")


def _run_func(
    func: Callable[..., R],
    acc: int,
    console: Console,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    results: deque[int | float] = deque(maxlen=acc)
    i: int = 0
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    with Live(
        Text.assemble("Run ", (str(i), "primary")),
        console=console,
        refresh_per_second=60,
    ) as live_display:
        for _ in repeat(None, acc):
            i += 1
            start_time: float = perf_counter_ns()
            func(*modded_args, **kwargs)
            end_time: float = perf_counter_ns()
            live_display.update(Text.assemble("Run ", (str(i), "primary")))
            results.append(end_time - start_time)
            if i == acc:
                live_display.refresh()
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
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    with Live(
        Text.assemble("Run ", (str(i), "primary")),
        console=console,
        refresh_per_second=60,
    ) as live_display:
        for _ in repeat(None, acc):
            i += 1
            start_time: float = perf_counter_ns()
            await func(*modded_args, **kwargs)
            end_time: float = perf_counter_ns()
            live_display.update(Text.assemble("Run ", (str(i), "primary")))
            results.append(end_time - start_time)
            if i == acc:
                live_display.refresh()
    return np.array(results)


def _visualize(func: Callable[..., R], acc: int, *args, **kwargs) -> np.ndarray:
    results: deque[int] = deque(maxlen=acc)
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    for _ in repeat(None, acc):
        start_time: float = perf_counter_ns()
        func(*modded_args, **kwargs)
        end_time: float = perf_counter_ns()
        results.append(end_time - start_time)
    return np.array(results)


async def _async_visualize(
    func: Callable[..., Awaitable[R]], acc: int, *args, **kwargs
) -> np.ndarray:
    results: deque[int] = deque(maxlen=acc)
    modded_args: list[Any] = []
    for arg in args:
        if isinstance(arg, SingleParam):
            modded_args.extend(arg)
        else:
            modded_args.append(arg)
    for _ in repeat(None, acc):
        start_time: float = perf_counter_ns()
        await func(*modded_args, **kwargs)
        end_time: float = perf_counter_ns()
        results.append(end_time - start_time)

    return np.array(results)


class Benchmark(Generic[P, R]):
    """
    A class for benchmarking synchronous functions by running them mulitple times
    and logging the result of the values using the method inputed
    """

    def __init__(
        self,
        func: Callable[P, R],
        precision: int = 15,
        unit: Literal["ns", "us", "ms", "s"] = "s",
        method: str = "trim_mean",
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[P, R]
            The  function to benchmark.
        precision : int, optional, default=15
            The number of times to run the function to get an average time.
        unit : Literal["ns", "us", "ms", "s"], optional, default="s"
            The unit of time that will be outputed.
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

        if method not in Constants.METHODS:
            raise ValueError(
                f"Method is not supported: {method}, (supported methods: {', '.join(Constants.METHODS.keys())})"
            )
        if not isinstance(func, Callable):
            raise TypeError("func must be of type Callable.")
        if not isinstance(precision, int):
            raise TypeError("precision must be of type int.")
        if precision < 1:
            raise ValueError("precision must be greater than or equal to 1.")
        if unit not in Constants.UNITS:
            raise ValueError(
                f"Unit does not exist: {unit}, (supported units: {', '.join(Constants.UNITS)})"
            )
        update_wrapper(self, func)
        wraps(func)(self)
        self.__method: str = method
        self.__func: Callable[P, R] = func
        self.__unit: Literal["ns", "us", "ms", "s"] = unit
        self.__precision: int = precision
        self.__results: np.ndarray
        self.__result: int | float
        self.__console: Console = GlobalConsole.get()

    def __repr__(self) -> str:
        return self.__func.__repr__()

    def benchmark_repr(self) -> str:
        return "".join(
            [
                "Benchmark",
                "(",
                f"func={self.__func.__name__}, ",
                f"precision={self.__precision}, ",
                f"unit={self.__unit!r}, ",
                f"method={self.__method!r}",
                ")",
            ]
        )

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
    def unit(self) -> Literal["ns", "us", "ms", "s"]:
        return self.__unit

    @unit.setter
    def unit(self, value: Literal["ns", "us", "ms", "s"]) -> None:
        if value not in Constants.UNITS:
            raise ValueError(
                f"Invalid Unit: {value}, Please use 'ns', 'us', 'ms', or 's'."
            )
        self.__unit = value

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        if value not in Constants.METHODS:
            raise ValueError(f"Invalid method: {value}")
        self.__method = value

    @property
    def func(self) -> Callable[P, R]:
        return self.__func

    @property
    def result(self) -> int | float | None:
        return self.__result

    def benchmark(self, *args: Any, **kwargs: Any) -> int | float:
        """Benchmarks the time of a function.

        Parameters
        ----------
        args : Any
            The arguments for the function.
        kwargs : Any
            The keyword arguments for the function.

        Returns
        -------
        time_taken_in_ns : int | float
            The time taken to run the function in nanoseconds.
        """

        self.__console.rule(
            Text.assemble(
                "Benchmarking ", _format_function(self.__func, False, *args, **kwargs)
            )
        )
        self.__results = _run_func(
            self.__func, self.__precision, self.__console, *args, **kwargs
        )
        current_locale: str = default_locale("LC_NUMERIC") or "en_US"
        self.__result = Constants.METHODS[self.__method](self.__results)
        unit: Decimal = Decimal(self.__result) / Decimal(
            1000 ** Constants.UNITS.index(self.__unit)
        )
        formatted: str = format_decimal(unit, locale=current_locale)
        expanded_unit: str = Constants.UNITS_DICT[self.__unit]
        if unit == 1:
            expanded_unit = expanded_unit[:-1]
        self.__console.log(
            Text.assemble(
                _format_function(self.__func, True, *args, **kwargs),
                f" took {formatted} {expanded_unit}",
            )
        )
        return self.__result

    def visual_benchmark(self, *args: Any, **kwargs: Any) -> BarContainer:
        """Benchmarks the time of a function and displays it in a bar graph.

        Parameters
        ----------
        args : Any
            The arguments for the function.
        kwargs : Any
            The keyword arguments for the function.

        Returns
        -------
        bar_plot : BarContainer
            The bar plot of each different run of the function.
        """

        self.__console.rule(
            Text.assemble(
                "Benchmarking ", _format_function(self.__func, False, *args, **kwargs)
            )
        )
        self.__results = _visualize(self.__func, self.__precision, *args, **kwargs)
        self.__result = Constants.METHODS[self.__method](self.__results)
        expanded_unit: str = Constants.UNITS_DICT[self.__unit]
        self.__fig, self.__ax = plt.subplots()
        self.__ax.set_xlabel("Run")
        self.__ax.set_ylabel(f"Time in {expanded_unit}")
        self.__fig.suptitle(f"{_format_function(self.__func, True, *args, **kwargs)}")
        return self.__ax.bar(
            [i for i in range(self.precision)],
            self.__results / (1000 ** Constants.UNITS.index(self.__unit)),
        )

    def show(self) -> None:
        """Shows the figure"""

        if not hasattr(self, f"_{self.__class__.__name__}__fig"):
            raise AttributeError("No graph has been created yet.")
        self.__fig.show()

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return self.__func(*args, **kwargs)

    def compare(
        self,
        func2: Callable[..., Any],
        args1: tuple[Any, ...] | SingleParam[Any] | None = None,
        args2: tuple[Any, ...] | SingleParam[Any] | None = None,
        kwargs1: dict[Any, Any] | None = None,
        kwargs2: dict[Any, Any] | None = None,
    ) -> None:
        """Compare the exectution time of two functions.

        Parameters
        ----------
        func2 : Callable[..., R]
            The second function to benchmark.
        args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict[Any, Any] | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict[Any, Any] | None, optional, default=None
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
        benchmark: Benchmark[P, R] = Benchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        time1: float = self.benchmark(*args1, **kwargs1)
        time2: float = benchmark.benchmark(*args2, **kwargs2)
        formatted_self: Text = _format_function(self.__func, True, *args1, **kwargs1)
        formatted_other: Text = _format_function(func2, True, *args2, **kwargs2)
        times_speed: str = (
            f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
        )
        self.__console.rule("Results")
        self.__console.log(
            Text.assemble(
                formatted_self,
                " is ",
                Text.assemble(times_speed, " times the faster than ", formatted_other),
            )
        )

    def visual_compare(
        self,
        func2: Callable[..., Any],
        args1: tuple[Any, ...] | SingleParam[Any] | None = None,
        args2: tuple[Any, ...] | SingleParam[Any] | None = None,
        kwargs1: dict[Any, Any] | None = None,
        kwargs2: dict[Any, Any] | None = None,
    ) -> None:
        """Compare the exectution time of two functions and display the results in
        two different bar graphs.

        Parameters
        ----------
        func2 : Callable[..., R]
            The second function to benchmark.
        args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict[Any, Any] | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict[Any, Any] | None, optional, default=None
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
        benchmark: Benchmark[P, R] = Benchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        self.visual_benchmark(*args1, **kwargs1)
        time1: float = cast(float, self.result)
        benchmark.visual_benchmark(*args2, **kwargs2)
        time2: float = cast(float, benchmark.result)
        formatted_self: Text = _format_function(self.__func, True, *args1, **kwargs1)
        formatted_other: Text = _format_function(func2, True, *args2, **kwargs2)
        times_speed: str = (
            f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
        )
        self.__console.rule("Results")
        self.__console.log(
            Text.assemble(
                formatted_self,
                " is ",
                Text.assemble(times_speed, " times the faster than ", formatted_other),
            )
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
        unit: Literal["ns", "us", "ms", "s"] = "s",
        method: str = "trim_mean",
    ) -> None:
        """
        Parameters
        ----------
        func : Callable[P, Awaitable[R]]
            The  function to benchmark.
        precision : int, optional, default=15
            The number of times to run the function to get an average time.
        unit : Literal["ns", "us", "ms", "s"], optional, default="s"
            The unit of time that will be outputed.
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

        if method not in Constants.METHODS:
            raise ValueError(
                f"Method is not supported: {method}, (supported methods: {', '.join(Constants.METHODS.keys())})"
            )
        if not isinstance(func, Callable):
            raise TypeError("func must be of type Callable.")
        if not isinstance(precision, int):
            raise TypeError("precision must be of type int.")
        if precision < 1:
            raise ValueError("precision must be greater than or equal to 1.")
        if unit not in Constants.UNITS:
            raise ValueError(
                f"Unit does not exist: {unit}, (supported units: {', '.join(Constants.UNITS)})"
            )
        update_wrapper(self, func)
        wraps(func)(self)
        self.__method: str = method
        self.__func: Callable[P, Awaitable[R]] = func
        self.__unit: Literal["ns", "us", "ms", "s"] = unit
        self.__precision: int = precision
        self.__results: np.ndarray
        self.__result: int | float
        self.__console: Console = GlobalConsole.get()

    def __repr__(self) -> str:
        return self.__func.__repr__()

    def benchmark_repr(self) -> str:
        return "".join(
            [
                "AsyncBenchmark",
                "(",
                f"func={self.__func.__name__}, ",
                f"precision={self.__precision}, ",
                f"unit={self.__unit!r}, ",
                f"method={self.__method!r}",
                ")",
            ]
        )

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
    def unit(self, value: Literal["ns", "us", "ms", "s"]) -> None:
        if value not in Constants.UNITS:
            raise ValueError(
                f"Invalid Unit: {value}, Please use 'ns', 'us', 'ms', or 's'."
            )
        self.__unit = value

    @property
    def method(self) -> str:
        return self.__method

    @method.setter
    def method(self, value: str) -> None:
        if value not in Constants.METHODS:
            raise ValueError(f"Invalid method: {value}")
        self.__method = value

    @property
    def func(self) -> Callable[P, Awaitable[R]]:
        return self.__func

    @property
    def result(self) -> int | float | None:
        return self.__result

    async def benchmark(self, *args: Any, **kwargs: Any) -> int | float:
        """Benchmarks the time of an asynchronous function.

        Parameters
        ----------
        args : Any
            The arguments for the function.
        kwargs : Any
            The keyword arguments for the function.

        Returns
        -------
        time_taken_in_ns : int | float
            The time taken to run the function in nanoseconds.
        """

        self.__console.rule(
            Text.assemble(
                "Benchmarking ", _format_function(self.__func, False, *args, **kwargs)
            )
        )
        self.__results = await _async_run_func(
            self.__func, self.__precision, self.__console, *args, **kwargs
        )
        current_locale: str = default_locale("LC_NUMERIC") or "en_US"
        self.__result = Constants.METHODS[self.__method](self.__results)
        unit: Decimal = Decimal(self.__result) / Decimal(
            1000 ** Constants.UNITS.index(self.__unit)
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
            Text.assemble(
                _format_function(self.__func, True, *args, **kwargs),
                f" took {formatted} {expanded_unit}",
            )
        )
        return self.__result

    async def visual_benchmark(self, *args: Any, **kwargs: Any) -> BarContainer:
        """Benchmarks the time of an asynchronous function and displays it in a bar graph.

        Parameters
        ----------
        args : Any
            The arguments for the function.
        kwargs : Any
            The keyword arguments for the function.

        Returns
        -------
        bar_plot : BarContainer
            The bar plot of each different run of the function.
        """

        self.__console.rule(
            Text.assemble(
                "Benchmarking ", _format_function(self.__func, False, *args, **kwargs)
            )
        )
        self.__results = await _async_visualize(
            self.__func, self.__precision, *args, **kwargs
        )
        self.__result = Constants.METHODS[self.__method](self.__results)
        expanded_unit: str = Constants.UNITS_DICT[self.__unit]
        self.__fig, self.__ax = plt.subplots()
        self.__ax.set_xlabel("Run")
        self.__ax.set_ylabel(f"Time in {expanded_unit}")
        self.__fig.suptitle(f"{_format_function(self.__func, True, *args, **kwargs)}")
        return self.__ax.bar(
            [i for i in range(self.precision)],
            self.__results / (1000 ** Constants.UNITS.index(self.__unit)),
        )

    def show(self) -> None:
        """Shows the figure"""

        if not hasattr(self, f"_{self.__class__.__name__}__fig"):
            raise AttributeError("No graph has been created yet.")
        self.__fig.show()

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        return await self.__func(*args, **kwargs)

    async def compare(
        self,
        func2: Callable[..., Awaitable[Any]],
        args1: tuple[Any, ...] | SingleParam[Any] | None = None,
        args2: tuple[Any, ...] | SingleParam[Any] | None = None,
        kwargs1: dict[Any, Any] | None = None,
        kwargs2: dict[Any, Any] | None = None,
    ) -> None:
        """Compare the exectution time of two functions.

        Parameters
        ----------
        func2 : Callable[..., Awaitable[Any]]
            The second asynchronous function to benchmark.
        args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict[Any, Any] | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict[Any, Any] | None, optional, default=None
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
        benchmark: AsyncBenchmark[P, R] = AsyncBenchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        time1: float = await self.benchmark(*args1, **kwargs1)
        time2: float = await benchmark.benchmark(*args2, **kwargs2)
        formatted_self: Text = _format_function(self.__func, True, *args1, **kwargs1)
        formatted_other: Text = _format_function(func2, True, *args2, **kwargs2)
        times_speed: str = (
            f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
        )
        self.__console.rule("Results")
        self.__console.log(
            Text.assemble(
                formatted_self,
                " is ",
                Text.assemble(times_speed, " times the faster than ", formatted_other),
            )
        )

    async def visual_compare(
        self,
        func2: Callable[..., Awaitable[Any]],
        args1: tuple[Any, ...] | SingleParam[Any] | None = None,
        args2: tuple[Any, ...] | SingleParam[Any] | None = None,
        kwargs1: dict[Any, Any] | None = None,
        kwargs2: dict[Any, Any] | None = None,
    ) -> None:
        """Compare the exectution time of two asynchronous functions and display the results in
        two different bar graphs.

        Parameters
        ----------
        func2 : Callable[..., R]
            The second asynchronous function to benchmark.
        args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the first function.
        args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
            The posistional arguments for the second function.
        kwargs1: dict[Any, Any] | None, optional, default=None
            The keyword arguments for the first function
        kwargs2: dict[Any, Any] | None, optional, default=None
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
        benchmark: AsyncBenchmark[P, R] = AsyncBenchmark(
            func2, self.__precision, self.__unit, self.__method
        )
        await self.visual_benchmark(*args1, **kwargs1)
        time1: float = cast(float, self.result)
        await benchmark.visual_benchmark(*args2, **kwargs2)
        time2: float = cast(float, benchmark.result)
        formatted_self: Text = _format_function(self.__func, True, *args1, **kwargs1)
        formatted_other: Text = _format_function(func2, True, *args2, **kwargs2)
        times_speed: str = (
            f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
        )
        self.__console.rule("Results")
        self.__console.log(
            Text.assemble(
                formatted_self,
                " is ",
                Text.assemble(times_speed, " times the faster than ", formatted_other),
            )
        )


async def compare(
    bench1: Benchmark | AsyncBenchmark,
    bench2: Benchmark | AsyncBenchmark,
    args1: tuple[Any, ...] | SingleParam[Any] | None = None,
    args2: tuple[Any, ...] | SingleParam[Any] | None = None,
    kwargs1: dict[Any, Any] | None = None,
    kwargs2: dict[Any, Any] | None = None,
) -> None:
    """Compare the exectution time of two benchmarks.
    Use this if you want to compare an async function with a synchronous function.

    Parameters
    ----------
    bench1 : Benchmark | AsyncBenchmark
        The first benchmark to compare.
    bench2 : Benchmark | AsyncBenchmark
        The second benchmark to compare.
    args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
        The posistional arguments for the first function.
    args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
        The posistional arguments for the second function.
    kwargs1: dict[Any, Any] | None, optional, default=None
        The keyword arguments for the first function
    kwargs2: dict[Any, Any] | None, optional, default=None
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
    console: Console = GlobalConsole.get()
    if isinstance(bench1, Benchmark):
        time1 = bench1.benchmark(*args1, **kwargs1)
    elif isinstance(bench1, AsyncBenchmark):
        time1 = await bench1.benchmark(*args1, **kwargs1)
    if isinstance(bench2, Benchmark):
        time2 = bench2.benchmark(*args2, **kwargs2)
    elif isinstance(bench2, AsyncBenchmark):
        time2 = await bench2.benchmark(*args2, **kwargs2)
    formatted_self: Text = _format_function(bench1.func, True, *args1, **kwargs1)
    formatted_other: Text = _format_function(bench2.func, True, *args2, **kwargs2)
    times_speed: str = f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
    console.rule("Results")
    console.log(
        Text.assemble(
            formatted_self,
            " is ",
            Text.assemble(times_speed, " times the faster than ", formatted_other),
        )
    )


async def visual_compare(
    bench1: Benchmark | AsyncBenchmark,
    bench2: Benchmark | AsyncBenchmark,
    args1: tuple[Any, ...] | SingleParam[Any] | None = None,
    args2: tuple[Any, ...] | SingleParam[Any] | None = None,
    kwargs1: dict[Any, Any] | None = None,
    kwargs2: dict[Any, Any] | None = None,
) -> None:
    """Compare the exectution time of two benchmarks and displays it in a bar graph.
    Use this if you want to compare an async function with a synchronous function.

    Parameters
    ----------
    bench1 : Benchmark | AsyncBenchmark
        The first benchmark to compare.
    bench2 : Benchmark | AsyncBenchmark
        The second benchmark to compare.
    args1 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
        The posistional arguments for the first function.
    args2 : tuple[Any, ...] | SingleParam[Any] | None, optional, default=None
        The posistional arguments for the second function.
    kwargs1: dict[Any, Any] | None, optional, default=None
        The keyword arguments for the first function
    kwargs2: dict[Any, Any] | None, optional, default=None
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
    console: Console = GlobalConsole.get()
    if isinstance(bench1, Benchmark):
        bench1.visual_benchmark(*args1, **kwargs1)
        time1 = cast(float, bench1.result)
    elif isinstance(bench1, AsyncBenchmark):
        await bench1.visual_benchmark(*args1, **kwargs1)
        time1 = cast(float, bench1.result)
    if isinstance(bench2, Benchmark):
        bench2.visual_benchmark(*args2, **kwargs2)
        time2 = cast(float, bench2.result)
    elif isinstance(bench2, AsyncBenchmark):
        await bench2.visual_benchmark(*args2, **kwargs2)
        time2 = cast(float, bench2.result)
    formatted_self: Text = _format_function(bench1.func, True, *args1, **kwargs1)
    formatted_other: Text = _format_function(bench2.func, True, *args2, **kwargs2)
    times_speed: str = f"{(time2 / time1) if time1 > 0 else ((time2 + 1) / time1):.4f}"
    console.rule("Results")
    console.log(
        Text.assemble(
            formatted_self,
            " is ",
            Text.assemble(times_speed, " times the faster than ", formatted_other),
        )
    )
