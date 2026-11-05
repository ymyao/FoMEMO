"""Benchmarks used in the paper's scalability and prior-mismatch studies."""

import numpy as np
from pymoo.problems.many import WFG4 as PymooWFG4

from .base import Benchmark


class WFG4(Benchmark):
    """Three-objective WFG4 with a configurable decision dimension."""

    def __init__(self, n_dim=30):
        upper = 2.0 * np.arange(1, n_dim + 1, dtype=np.float64)
        super().__init__(f"WFG4 ({n_dim}d3m)", n_dim, 3, np.zeros(n_dim), upper)
        self._problem = PymooWFG4(n_var=n_dim, n_obj=3)

    def _evaluate_normalized(self, x):
        return self._problem.evaluate(self.to_physical(x))


def _triangle_wave(x):
    return np.abs(x - np.round(x))


def _takagi(x, n_terms=20):
    result = np.zeros_like(x, dtype=np.float64)
    for index in range(n_terms):
        result += 0.5**index * _triangle_wave(2**index * x)
    return result


def _weierstrass(x, a=0.7, b=3.0, n_terms=20):
    result = np.zeros_like(x, dtype=np.float64)
    for index in range(n_terms):
        result += a**index * np.cos(b**index * np.pi * x)
    return result


class WeierstrassTakagi(Benchmark):
    """One-dimensional non-smooth problem on the physical domain [0, 0.5]."""

    def __init__(self):
        super().__init__(
            "Weierstrass-Takagi",
            1,
            2,
            lower_bounds=[0.0],
            upper_bounds=[0.5],
        )

    def _evaluate_normalized(self, x):
        physical = self.to_physical(x)
        first = _weierstrass(physical).mean(axis=1)
        second = _takagi(physical).mean(axis=1)
        normalized = np.column_stack(
            [
                (first + 1.84) / 5.18,
                second / 0.67,
            ]
        )
        return np.clip(normalized, 0.0, 1.0)
