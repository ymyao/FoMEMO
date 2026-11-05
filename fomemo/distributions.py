"""Predictive distributions and training loss for FoMEMO."""

import torch
from torch import nn


class FullSupportBarDistribution(nn.Module):
    """Piecewise distribution with half-normal tails."""

    def __init__(self, borders, ignore_nan_targets=True):
        super().__init__()
        if borders.ndim != 1 or len(borders) < 3:
            raise ValueError("At least three one-dimensional borders are required.")
        self.register_buffer("borders", borders)
        self.register_buffer("smoothing", torch.tensor(0.0))
        self.register_buffer("bucket_widths", borders[1:] - borders[:-1])
        if not (self.bucket_widths >= 0).all() or not (borders[-1] > borders[0]):
            raise ValueError("Borders must be sorted and span a positive range.")
        self.num_bars = len(borders) - 1
        self.ignore_nan_targets = ignore_nan_targets
        self.to(borders.device)

    def forward(self, logits, y):
        """Negative log density, including the two half-normal tails."""
        self.to(y.device)
        y = y.clone().view(len(y), -1)
        ignored = torch.isnan(y)
        if ignored.any() and not self.ignore_nan_targets:
            raise ValueError("Found NaN in training targets.")
        y[ignored] = self.borders[0]
        indices = torch.searchsorted(self.borders, y) - 1
        indices[y == self.borders[0]] = 0
        indices[y == self.borders[-1]] = self.num_bars - 1
        indices.clamp_(0, self.num_bars - 1)
        if logits.shape[-1] != self.num_bars:
            raise ValueError("Logit count does not match the number of buckets.")
        scaled = torch.log_softmax(logits, -1) - torch.log(self.bucket_widths)
        log_probs = scaled.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        left, right = indices == 0, indices == self.num_bars - 1
        left_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[0])
        right_normal = self.halfnormal_with_p_weight_before(self.bucket_widths[-1])
        log_probs[left] += left_normal.log_prob((self.borders[1] - y[left]).clamp(min=1e-8)) + torch.log(self.bucket_widths[0])
        log_probs[right] += right_normal.log_prob((y[right] - self.borders[-2]).clamp(min=1e-8)) + torch.log(self.bucket_widths[-1])
        loss = -log_probs
        if self.smoothing:
            smoothing = self.smoothing if self.training else 0.0
            loss = (1 - smoothing) * loss - smoothing * scaled.mean(dim=-1)
        if ignored.any():
            loss[ignored] = 0.0
        return loss

    def __setstate__(self, state):
        super().__setstate__(state)
        self.__dict__.setdefault("append_mean_pred", False)

    @staticmethod
    def halfnormal_with_p_weight_before(range_max, p=0.5):
        scale = range_max / torch.distributions.HalfNormal(
            torch.tensor(1.0, device=range_max.device)
        ).icdf(torch.tensor(p, device=range_max.device))
        return torch.distributions.HalfNormal(scale)

    def icdf(self, logits, left_prob):
        probabilities = logits.softmax(-1)
        cumulative = torch.cumsum(probabilities, dim=-1)
        requested = left_prob * torch.ones(
            *cumulative.shape[:-1],
            1,
            device=logits.device,
        )
        index = torch.searchsorted(cumulative, requested).squeeze(-1)
        index = index.clamp(0, cumulative.shape[-1] - 1)
        cumulative = torch.cat(
            [
                torch.zeros(
                    *cumulative.shape[:-1],
                    1,
                    device=logits.device,
                ),
                cumulative,
            ],
            dim=-1,
        )
        residual = left_prob - cumulative.gather(
            -1,
            index[..., None],
        ).squeeze(-1)
        left = self.borders[index]
        right = self.borders[index + 1]
        selected_probability = probabilities.gather(
            -1,
            index[..., None],
        ).squeeze(-1)
        return left + (right - left) * residual / selected_probability

    def ucb(self, logits, best_f=None, rest_prob=(1 - 0.682) / 2, maximize=True):
        probability = 1 - rest_prob if maximize else rest_prob
        return self.icdf(logits, probability)

    def mean(self, logits):
        bucket_means = self.borders[:-1] + self.bucket_widths / 2
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        bucket_means = bucket_means.clone()
        bucket_means[0] = -side_normals[0].mean + self.borders[1]
        bucket_means[-1] = side_normals[1].mean + self.borders[-2]
        return torch.softmax(logits, -1) @ bucket_means.to(logits.device)

    def mean_of_square(self, logits):
        left = self.borders[:-1]
        right = self.borders[1:]
        values = (left.square() + right.square() + left * right) / 3
        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        values = values.clone()
        values[0] = (
            side_normals[0].variance
            + (-side_normals[0].mean + self.borders[1]).square()
        )
        values[-1] = (
            side_normals[1].variance
            + (side_normals[1].variance + self.borders[-2]).square()
        )
        return torch.softmax(logits, -1) @ values

    def variance(self, logits):
        return self.mean_of_square(logits) - self.mean(logits).square()

    @staticmethod
    def ei_for_halfnormal(scale, best_f):
        standardized = -best_f / scale
        normal = torch.distributions.Normal(
            torch.zeros_like(standardized),
            torch.ones_like(standardized),
        )
        normal_ei = scale * (
            torch.exp(normal.log_prob(standardized))
            + standardized * normal.cdf(standardized)
        )
        return 2 * normal_ei

    def ei(self, logits, best_f, maximize=True):
        if torch.isnan(logits).any():
            raise ValueError("Model logits contain NaN values.")
        if not maximize:
            raise ValueError("FoMEMO scalar utilities are maximized.")
        if not torch.is_tensor(best_f) or not len(best_f.shape):
            best_f = torch.full(
                logits[..., 0].shape,
                best_f,
                device=logits.device,
            )
        if best_f.shape != logits[..., 0].shape:
            raise ValueError("best_f and logits have incompatible shapes.")

        bucket_diffs = self.borders[1:] - self.borders[:-1]
        expanded_best = best_f[..., None].repeat(
            *[1] * len(best_f.shape),
            logits.shape[-1],
        )
        clamped_best = expanded_best.clamp(
            self.borders[:-1],
            self.borders[1:],
        )
        contributions = (
            (self.borders[1:].square() - clamped_best.square()) / 2
            - expanded_best * (self.borders[1:] - clamped_best)
        ) / bucket_diffs

        side_normals = (
            self.halfnormal_with_p_weight_before(self.bucket_widths[0]),
            self.halfnormal_with_p_weight_before(self.bucket_widths[-1]),
        )
        side_positions = (
            -(best_f - self.borders[1]).clamp(max=0.0),
            (best_f - self.borders[-2]).clamp(min=0.0),
        )
        contributions[..., -1] = self.ei_for_halfnormal(
            side_normals[1].scale,
            side_positions[1],
        )
        contributions[..., 0] = self.ei_for_halfnormal(
            side_normals[0].scale,
            torch.zeros_like(side_positions[0]),
        ) - self.ei_for_halfnormal(
            side_normals[0].scale,
            side_positions[0],
        )
        return torch.einsum(
            "...b,...b->...",
            torch.softmax(logits, -1),
            contributions,
        )
