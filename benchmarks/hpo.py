"""Table-based 3D reconstruction hyperparameter benchmarks."""

import csv
from functools import lru_cache
from pathlib import Path

import numpy as np

from .base import Benchmark


_FEATURES = (
    "sh_degree",
    "lambda_dssim",
    "percent_dense",
    "scale_pos_scale_lr",
    "scale_feat_opacity_lr",
)
_DATA_PATH = Path(__file__).with_name("data") / "record_synthetic_v2.csv"


@lru_cache(maxsize=1)
def _load_rows():
    rows = []
    with _DATA_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "scene": row["scene"],
                    "features": [float(row[name]) for name in _FEATURES],
                    "psnr": float(row["psnr"]),
                    "size": float(row["size"]),
                    "npt": float(row["npt"]),
                }
            )
    return tuple(rows)


def _minmax(values):
    minimum = values.min(axis=0)
    width = values.max(axis=0) - minimum
    if np.any(width == 0.0):
        raise ValueError("A benchmark data column has zero range.")
    return (values - minimum) / width


class ReconstructionHPO(Benchmark):
    def __init__(self, scene, n_obj):
        display = "Materials" if scene == "materials" else scene.capitalize()
        super().__init__(display, n_dim=5, n_obj=n_obj)
        selected = [row for row in _load_rows() if row["scene"] == scene]
        if not selected:
            raise ValueError(f"No records found for scene: {scene}")
        self.scene = scene
        self._features = _minmax(
            np.asarray([row["features"] for row in selected], dtype=np.float64)
        )
        psnr = _minmax(
            np.asarray([row["psnr"] for row in selected], dtype=np.float64)[:, None]
        )[:, 0]
        size = _minmax(
            np.asarray([row["size"] for row in selected], dtype=np.float64)[:, None]
        )[:, 0]
        npt = _minmax(
            np.asarray([row["npt"] for row in selected], dtype=np.float64)[:, None]
        )[:, 0]
        columns = [1.0 - psnr, size]
        if n_obj == 3:
            columns.append(npt)
        self._objectives = np.column_stack(columns)

    def _evaluate_normalized(self, x):
        output = np.empty((len(x), self.n_obj), dtype=np.float64)
        for index, row in enumerate(x):
            distance = np.linalg.norm(self._features - row, axis=1)
            output[index] = self._objectives[np.argmin(distance)]
        return output


def make_lego():
    return ReconstructionHPO("lego", 2)


def make_materials():
    return ReconstructionHPO("materials", 2)


def make_mic():
    return ReconstructionHPO("mic", 3)


def make_ship():
    return ReconstructionHPO("ship", 3)
