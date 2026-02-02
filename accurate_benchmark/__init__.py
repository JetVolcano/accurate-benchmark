"""
Accurate Benchmark
------------------
This is a python package for accurate benchmarking and speed comparisons

"""

from accurate_benchmark.benchmarking import AsyncBenchmark, Benchmark
from accurate_benchmark.parameters import SingleParam

from . import benchmarking, parameters

__all__: list[str] = [
    "AsyncBenchmark",
    "Benchmark",
    "SingleParam",
    "benchmarking",
    "parameters",
]
__version__: str = "1.4.3"
