from collections.abc import Callable
from functools import partial
from typing import Any, Final

import numpy as np
from scipy.stats import trim_mean

from accurate_benchmark._console import _BenchmarkConsole


class Constant(type):
    def __setattr__(cls, name: str, value: Any) -> None:
        if name in cls.__dict__:
            raise AttributeError(f"Cannot modify constant: '{name}'")
        super().__setattr__(name, value)


class Constants(metaclass=Constant):
    GLOBAL_CONSOLE: Final[_BenchmarkConsole] = _BenchmarkConsole()
    UNITS: Final[tuple[str, str, str, str]] = ("ns", "us", "ms", "s")
    UNITS_DICT: Final[dict[str, str]] = {
        "ns": "nanoseconds",
        "us": "microseconds",
        "ms": "milliseconds",
        "s": "seconds",
    }
    METHODS: Final[dict[str, Callable[..., float]]] = {
        "trim_mean": partial(trim_mean, proportiontocut=0.05),
        "mean": np.mean,
        "median": np.median,
    }
