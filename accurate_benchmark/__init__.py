"""
Accurate Benchmark
------------------
This is a python package for accurate benchmarking and speed comparisons

"""

from accurate_benchmark.benchmarking import Benchmark
from accurate_benchmark.parameters import SingleParam

__all__: list[str] = ["Benchmark", "SingleParam", "benchmarking", "parameters"]
__version__: str = "1.4.3"
