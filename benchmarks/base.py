"""Shared interface for the benchmark problems."""

from abc import ABC, abstractmethod

import numpy as np


class Benchmark(ABC):
    """A minimization benchmark evaluated from normalized decision variables."""

    def __init__(self, name, n_dim, n_obj, lower_bounds=None, upper_bounds=None):
        self.name = name
        self.n_dim = int(n_dim)
        self.n_obj = int(n_obj)
        self.lower_bounds = (
            np.zeros(self.n_dim, dtype=np.float64)
            if lower_bounds is None
            else np.asarray(lower_bounds, dtype=np.float64)
        )
        self.upper_bounds = (
            np.ones(self.n_dim, dtype=np.float64)
            if upper_bounds is None
            else np.asarray(upper_bounds, dtype=np.float64)
        )
        if self.lower_bounds.shape != (self.n_dim,):
            raise ValueError("lower_bounds has the wrong shape.")
        if self.upper_bounds.shape != (self.n_dim,):
            raise ValueError("upper_bounds has the wrong shape.")

    def evaluate(self, x):
        """Evaluate an ``(n, n_dim)`` array whose entries lie in ``[0, 1]``."""
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.n_dim:
            raise ValueError(
                f"{self.name} expects x with shape (n, {self.n_dim}); "
                f"received {x.shape}."
            )
        if not np.isfinite(x).all():
            raise ValueError("Decision variables must be finite.")
        tolerance = 1e-12
        if np.any(x < -tolerance) or np.any(x > 1.0 + tolerance):
            raise ValueError("Decision variables must be normalized to [0, 1].")
        x = np.clip(x, 0.0, 1.0)
        values = np.asarray(self._evaluate_normalized(x), dtype=np.float64)
        if values.shape != (len(x), self.n_obj):
            raise RuntimeError(
                f"{self.name} returned {values.shape}; expected "
                f"({len(x)}, {self.n_obj})."
            )
        if not np.isfinite(values).all():
            raise RuntimeError(f"{self.name} produced a non-finite objective value.")
        return values

    def to_physical(self, x):
        """Map normalized decisions to the benchmark's original domain."""
        x = np.asarray(x, dtype=np.float64)
        return self.lower_bounds + x * (self.upper_bounds - self.lower_bounds)

    @abstractmethod
    def _evaluate_normalized(self, x):
        raise NotImplementedError


class RowWisePhysicalBenchmark(Benchmark):
    """Base class for formulae defined one physical design at a time."""

    def _evaluate_normalized(self, x):
        physical_x = self.to_physical(x)
        return np.vstack([self._evaluate_physical(row) for row in physical_x])

    @abstractmethod
    def _evaluate_physical(self, x):
        raise NotImplementedError
