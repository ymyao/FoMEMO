"""Performance metrics for FoMEMO benchmark experiments."""

from pathlib import Path

import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from pymoo.indicators.hv import Hypervolume
from pymoo.indicators.igd_plus import IGDPlus

from benchmarks.engineering import CarCabDesign
from benchmarks.extensions import WFG4, WeierstrassTakagi
from benchmarks.hpo import ReconstructionHPO
from benchmarks.synthetic import CompositeSynthetic


def _make_metric(problem, metric_name, device):
    normalized = isinstance(
        problem, (CompositeSynthetic, ReconstructionHPO, WeierstrassTakagi)
    )
    if metric_name not in ("hv", "hv2", "igd+"):
        raise ValueError(f"Unknown metric: {metric_name!r}.")
    if metric_name == "hv2" and not normalized:
        raise ValueError(f"hv2 has no reference point for {problem.name}.")
    if metric_name == "igd+" and not isinstance(problem, (WFG4, CarCabDesign)):
        raise ValueError(f"igd+ has no reference Pareto front for {problem.name}.")

    if isinstance(problem, WFG4):
        # Match the reference project's WFG Pareto-front sampling defaults.
        front = problem._problem.pareto_front(
            n_iterations=200, points_each_iteration=1000
        )
        ideal, nadir = front.min(axis=0), front.max(axis=0)
    elif normalized:
        ideal, nadir = np.zeros(problem.n_obj), np.ones(problem.n_obj)
    else:
        ideal, nadir = problem.ideal_point, problem.nadir_point

    if metric_name == "igd+" and isinstance(problem, CarCabDesign):
        front = np.loadtxt(
            Path(__file__).resolve().parents[1] / "benchmarks" / "data" / "reference_points_RE91.dat",
            dtype=np.float64,
        )

    metric = (
        IGDPlus(front)
        if metric_name == "igd+"
        else Hypervolume(ref_point=np.full(problem.n_obj, 1.1))
    )
    ideal = torch.as_tensor(ideal, dtype=torch.float64, device=device)
    nadir = torch.as_tensor(nadir, dtype=torch.float64, device=device)

    def score(train_y):
        pareto_y = train_y[is_non_dominated(-train_y)]
        if metric_name == "hv":
            pareto_y = (pareto_y - ideal) / (nadir - ideal)
        return float(metric.do(pareto_y.detach().cpu().numpy()))

    return score, ideal, nadir
