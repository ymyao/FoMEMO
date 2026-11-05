"""FoMEMO candidate generation."""

import math
from pathlib import Path

import numpy as np
import scipy.optimize
import torch
from botorch.utils.sampling import draw_sobol_samples
from pymoo.util.reference_direction import sample_on_unit_simplex

from .checkpoint import load_model


eps = 0.1 # epsilon away from (0,0) ideal point

_ACQUISITIONS = {"ei", "ucb", "uhvi", "ur2i"}


def _decomposition_metric(values, preferences):
    weighted = values.transpose(0, 1) * preferences
    return -weighted.max(dim=2).values.transpose(0, 1)


def _sphere_preferences(count, objectives):
    weights = torch.abs(torch.randn(count, objectives))
    weights /= torch.norm(weights, dim=1).unsqueeze(1)
    weights = torch.clamp(weights, min=1e-12)
    inverse = 1 / weights
    return inverse / inverse.sum(dim=1, keepdim=True)


class FoMEMOOptimizer:
    """Generate FoMEMO candidates for normalized continuous search spaces."""

    def __init__(
        self,
        model_path="fomemo_model.pt",
        device="auto",
        *,
        num_random_samples=1024,
        num_restarts=20,
        num_grad_steps=200,
        num_preferences=100,
    ):
        self.device = self._resolve_device(device)
        self.model_path = Path(model_path).expanduser()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        self.num_random_samples = self._positive_int(
            num_random_samples,
            "num_random_samples",
        )
        self.num_restarts = self._positive_int(num_restarts, "num_restarts")
        self.num_grad_steps = self._positive_int(
            num_grad_steps,
            "num_grad_steps",
        )
        self.num_preferences = self._positive_int(
            num_preferences,
            "num_preferences",
        )
        if self.num_restarts > self.num_random_samples:
            raise ValueError("num_restarts cannot exceed num_random_samples.")
        self.model = load_model(self.model_path, self.device)

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return resolved

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
        return int(value)

    def _prepare_data(self, train_x, train_y):
        train_x = torch.as_tensor(
            train_x,
            dtype=torch.float64,
            device=self.device,
        )
        train_y = torch.as_tensor(
            train_y,
            dtype=torch.float64,
            device=self.device,
        )
        if train_x.ndim != 2 or train_y.ndim != 2:
            raise ValueError("train_x and train_y must both be two-dimensional.")
        if train_x.shape[0] != train_y.shape[0]:
            raise ValueError("train_x and train_y must have the same number of rows.")
        if not 1 <= train_x.shape[1] <= 50:
            raise ValueError("FoMEMO supports between 1 and 50 decision variables.")
        if not 2 <= train_y.shape[1] <= 10:
            raise ValueError("FoMEMO supports between 2 and 10 objectives.")
        if not torch.isfinite(train_x).all() or not torch.isfinite(train_y).all():
            raise ValueError("Training data must contain only finite values.")
        tolerance = 1e-7
        if (train_x < -tolerance).any() or (train_x > 1 + tolerance).any():
            raise ValueError("train_x must be normalized to [0, 1].")
        train_x = train_x.clamp(0.0, 1.0)
        minimum = train_y.min(dim=0, keepdim=True).values
        maximum = train_y.max(dim=0, keepdim=True).values
        normalized_y = (train_y - minimum) / torch.clamp(
            maximum - minimum,
            min=1e-12,
        )
        scaled_y = normalized_y * (1 - 2 * eps) + eps # rescaled to [eps,1-eps]
        return train_x, scaled_y

    def _model_ucb(self, train_x, train_y, test_x, preferences):
        preference_count = preferences.size(0)
        logits = self.model(
            train_x.unsqueeze(1).expand(-1, preference_count, -1).float(),
            train_y.unsqueeze(1).expand(-1, preference_count, -1).float(),
            test_x.unsqueeze(1).expand(-1, preference_count, -1).float(),
            style=preferences.float(),
        )
        objective_count = train_y.shape[1]
        return self.model.criterion[objective_count].ucb(
            logits,
            best_f=None,
            rest_prob=(1 - 0.682) / 2,
        )

    def _ei(self, train_x, train_y, test_x, preference):
        logits = self.model(
            train_x.unsqueeze(1).float(),
            train_y.unsqueeze(1).float(),
            test_x.unsqueeze(1).float(),
            style=preference.float(),
        )
        target = _decomposition_metric(
            train_y.unsqueeze(1),
            preference.unsqueeze(1),
        ).float().squeeze(1)
        return self.model.criterion[train_y.shape[1]].ei(
            logits,
            best_f=max(target),
        ).squeeze()

    def _ucb(self, train_x, train_y, test_x, preference):
        return self._model_ucb(
            train_x,
            train_y,
            test_x,
            preference,
        ).squeeze()

    def _uhvi(self, train_x, train_y, test_x):
        objective_count = train_y.shape[1]
        constant = math.pi ** (objective_count / 2) / (
            2**objective_count * math.gamma(objective_count / 2 + 1)
        )
        preferences = _sphere_preferences(
            self.num_preferences,
            objective_count,
        ).to(self.device)
        coefficient = torch.sum(1 / preferences.pow(2), dim=1).sqrt()
        augmented_y = torch.cat(
            [train_y, torch.eye(objective_count, device=self.device)],
            dim=0,
        )
        scalar_train = -_decomposition_metric(
            augmented_y.unsqueeze(1).expand(
                -1,
                self.num_preferences,
                -1,
            ),
            preferences.unsqueeze(1),
        )
        scalar_train = scalar_train.min(dim=0).values
        scalar_test = -self._model_ucb(
            train_x,
            augmented_y,
            test_x,
            preferences,
        )
        improvement = torch.clamp(
            scalar_train.pow(objective_count)
            - scalar_test.pow(objective_count),
            min=0.0,
        )
        piecewise = coefficient.pow(objective_count) * improvement
        return constant * piecewise.mean(dim=1)

    def _ur2i(self, train_x, train_y, test_x):
        objective_count = train_y.shape[1]
        directions = sample_on_unit_simplex(
            n_points=self.num_preferences,
            n_dim=objective_count,
            unit_simplex_mapping="das-dennis",
        )
        preferences = torch.tensor(directions, device=self.device)
        preference_count = preferences.shape[0]
        scalar_train = -_decomposition_metric(
            train_y.unsqueeze(1).expand(-1, preference_count, -1),
            preferences.unsqueeze(1),
        )
        scalar_train = scalar_train.min(dim=0).values
        scalar_test = -self._model_ucb(
            train_x,
            train_y,
            test_x,
            preferences,
        )
        return torch.clamp(
            scalar_train - scalar_test,
            min=0.0,
        ).mean(dim=1)

    def _acquisition(self, train_x, train_y, test_x, preference, name):
        if name == "ei":
            return self._ei(train_x, train_y, test_x, preference)
        if name == "ucb":
            return self._ucb(train_x, train_y, test_x, preference)
        if name == "uhvi":
            return self._uhvi(train_x, train_y, test_x)
        return self._ur2i(train_x, train_y, test_x)

    def _optimize_one(self, train_x, train_y, preference, acquisition):
        dimension = train_x.shape[1]
        bounds = torch.tensor(
            [[0.0] * dimension, [1.0] * dimension]
        )
        initial = draw_sobol_samples(
            bounds=bounds,
            n=self.num_random_samples,
            q=1,
        ).squeeze(1).to(self.device)
        with torch.no_grad():
            values = self._acquisition(
                train_x,
                train_y,
                initial,
                preference,
                acquisition,
            )
            restart_indices = values.argsort(descending=True)[
                : self.num_restarts
            ].cpu()
            scipy_initial = (
                initial[restart_indices]
                .detach()
                .cpu()
                .flatten()
                .numpy()
            )

        def objective(flat_x):
            evaluated = (
                torch.tensor(flat_x)
                .view(-1, dimension)
                .float()
                .to(self.device)
                .requires_grad_(True)
            )
            values = self._acquisition(
                train_x,
                train_y,
                evaluated,
                preference,
                acquisition,
            )
            loss = -values.mean()
            loss.backward()
            with torch.no_grad():
                evaluated.grad[
                    evaluated.grad != evaluated.grad
                ] = 0.0
            return (
                loss.detach().cpu().to(torch.float64).numpy(),
                evaluated.grad.detach()
                .view(*flat_x.shape)
                .cpu()
                .to(torch.float64)
                .numpy(),
            )

        result = scipy.optimize.minimize(
            objective,
            scipy_initial,
            method="L-BFGS-B",
            jac=True,
            bounds=[(0, 1)] * scipy_initial.size,
            options={"maxiter": self.num_grad_steps},
        )
        candidates = (
            torch.tensor(result.x)
            .float()
            .view(-1, dimension)
            .clamp(0.0, 1.0)
        )
        with torch.no_grad():
            values = self._acquisition(
                train_x,
                train_y,
                candidates.to(self.device),
                preference,
                acquisition,
            )
        return candidates[values.argmax().item()].to(self.device).detach()

    def suggest(self, train_x, train_y, batch_size=1, acquisition="ucb"):
        """Return normalized candidates with shape ``(batch_size, n_dim)``."""
        if acquisition not in _ACQUISITIONS:
            choices = ", ".join(sorted(_ACQUISITIONS))
            raise ValueError(f"acquisition must be one of: {choices}.")
        batch_size = self._positive_int(batch_size, "batch_size")
        train_x, train_y = self._prepare_data(train_x, train_y)
        objective_count = train_y.shape[1]
        candidates = []
        for _ in range(batch_size):
            preference = torch.distributions.Dirichlet(
                torch.ones(objective_count, device=self.device)
            ).sample((1,))
            candidates.append(
                self._optimize_one(
                    train_x,
                    train_y,
                    preference,
                    acquisition,
                )
            )
        return torch.stack(candidates).detach().cpu()
