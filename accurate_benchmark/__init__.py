"""
# Accurate Benchmark

A python package for accurate benchmarking and speed comparisons

## Features

- Asynchronous Benchmarking
- Comparing asynchronous functions with synchronous ones
- Applying a decorator to any function to allow it to be benchmarked without creating a different variable for the function
- A tool for visualizing the benchmark results

## Requirements / Dependencies

- Python **3.13 or higher**
- [matplotlib>=3.10.8](https://pypi.org/project/matplotlib/)
- [numpy>=2.4.2](https://pypi.org/project/numpy/)
- [rich>=14.3.2](https://pypi.org/project/rich/)
- [scipy>=1.17.0](https://pypi.org/project/scipy/)

"""

from accurate_benchmark._constants import Constants
from accurate_benchmark.benchmarking import (
    AsyncBenchmark,
    Benchmark,
    compare,
    visual_compare,
)
from accurate_benchmark.parameters import SingleParam

from . import benchmarking, parameters


__all__: list[str] = [
    "AsyncBenchmark",
    "Benchmark",
    "Constants",
    "SingleParam",
    "benchmarking",
    "compare",
    "parameters",
    "visual_compare",
]
__version__: str = "4.0.0"
