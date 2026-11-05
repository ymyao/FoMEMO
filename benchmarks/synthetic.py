"""Synthetic black-box functions used in the paper."""

import math

import numpy as np

from .base import Benchmark


def _normalize(value, maximum, minimum):
    if np.isinf(value) or value > maximum:
        return 1.0
    if np.isnan(value) or value < minimum:
        return 0.0
    return (value - minimum) / (maximum - minimum)


def _branin(x):
    scaled = 10.0 * x
    x1, x2 = scaled
    b = 5.1 / (4.0 * math.pi**2)
    c = 5.0 / math.pi
    value = (
        (x2 - b * x1**2 + c * x1 - 6.0) ** 2
        + 10.0 * (1.0 - 1.0 / (8.0 * math.pi)) * math.cos(x1)
        + 10.0
    )
    return _normalize(-value, 0.0, -100.0)


def _currin(x):
    x1, x2 = x
    attenuation = 1.0 if x2 == 0.0 else 1.0 - math.exp(-0.5 / x2)
    numerator = 2300.0 * x1**3 + 1900.0 * x1**2 + 2092.0 * x1 + 60.0
    denominator = 100.0 * x1**3 + 500.0 * x1**2 + 4.0 * x1 + 20.0
    value = attenuation * numerator / denominator
    return _normalize(-value, 0.0, -14.0)


def _ackley(x):
    x1, x2 = -30.0 + 60.0 * x
    square_sum = x1**2 + x2**2
    cosine_sum = math.cos(2.0 * math.pi * x1) + math.cos(2.0 * math.pi * x2)
    value = (
        -20.0 * math.exp(-0.2 * math.sqrt(0.5 * square_sum))
        - math.exp(0.5 * cosine_sum)
        + 20.0
        + math.e
    )
    return _normalize(value, 28.0, 0.0)


def _rosenbrock(x):
    x1, x2 = -2.0 + 4.0 * x
    value = (x1 - 1.0) ** 2 + 100.0 * (x2 - x1**2) ** 2
    return _normalize(value, 4400.0, 0.0)


def _dixon_price(x):
    x1, x2 = -10.0 + 20.0 * x
    value = (x1 - 1.0) ** 2 + 2.0 * (2.0 * x2**2 - x1) ** 2
    return _normalize(value, 88311.0, 0.0)


class CompositeSynthetic(Benchmark):
    def __init__(self, name, objectives):
        super().__init__(name=name, n_dim=2, n_obj=len(objectives))
        self._objectives = tuple(objectives)

    def _evaluate_normalized(self, x):
        return np.asarray(
            [[objective(row) for objective in self._objectives] for row in x],
            dtype=np.float64,
        )


def make_bc():
    return CompositeSynthetic("BC (Branin-Currin)", (_branin, _currin))


def make_ar():
    return CompositeSynthetic("AR (Ackley-Rosenbrock)", (_ackley, _rosenbrock))


def make_abc():
    return CompositeSynthetic(
        "ABC (Ackley-Branin-Currin)",
        (_ackley, _branin, _currin),
    )


def make_abd():
    return CompositeSynthetic(
        "ABD (Ackley-Branin-Dixon)",
        (_ackley, _branin, _dixon_price),
    )


def make_acd():
    return CompositeSynthetic(
        "ACD (Ackley-Currin-Dixon)",
        (_ackley, _currin, _dixon_price),
    )


def make_bcd():
    return CompositeSynthetic(
        "BCD (Branin-Currin-Dixon)",
        (_branin, _currin, _dixon_price),
    )
