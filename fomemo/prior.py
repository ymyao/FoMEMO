"""Gaussian-process prior used to train FoMEMO."""

import random
from dataclasses import dataclass

import gpytorch
import torch

from .config import DATA_CONFIG


@dataclass
class Batch:
    x: torch.Tensor
    y: torch.Tensor
    target_y: torch.Tensor
    style: torch.Tensor
    single_eval_pos: int = None


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x, likelihood):
        super().__init__(x, torch.Tensor(), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.constant = DATA_CONFIG["gp_mean"]
        kernel = DATA_CONFIG["gp_kernel"](ard_num_dims=x.size(-1), batch_shape=(x.size(0),))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel, batch_shape=(x.size(0),))

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, num_objs, device):
    """Sample independent GP objectives and a Dirichlet preference per sequence."""
    with gpytorch.settings.fast_computations(*DATA_CONFIG["gp_fast_computations"]):
        x = torch.rand(batch_size, seq_len, num_features, device=device)
        with gpytorch.settings.prior_mode(True):
            sample = torch.zeros(seq_len, batch_size, num_objs, device=device)
            for i in range(num_objs):
                likelihood = gpytorch.likelihoods.GaussianLikelihood(
                    noise_constraint=gpytorch.constraints.GreaterThan(DATA_CONFIG["gp_noise_lower_bound"])
                )
                model = ExactGPModel(x, likelihood)
                model.likelihood.noise = torch.ones_like(model.likelihood.noise) * DATA_CONFIG["gp_noise_variance"]
                model.covar_module.outputscale = torch.ones_like(model.covar_module.outputscale) * DATA_CONFIG["gp_output_scale"]
                prior = gpytorch.priors.GammaPrior(
                    torch.tensor(float(DATA_CONFIG["lengthscale_concentration"]), device=device),
                    torch.tensor(float(DATA_CONFIG["lengthscale_rate"]), device=device),
                )
                model.covar_module.base_kernel.lengthscale = prior.rsample((batch_size, num_features))
                model.to(device)
                sample[:, :, i] = likelihood(model(x)).sample().transpose(0, 1)
            concentration = torch.ones(num_objs, device=device) * DATA_CONFIG["preference_concentration"]
            preference = torch.distributions.Dirichlet(concentration).sample((batch_size,)).unsqueeze(1)
            y = torch.sigmoid(DATA_CONFIG["sigmoid_scale"] * sample)
            target = -(y.transpose(0, 1) * preference).max(dim=2).values.transpose(0, 1)
    return Batch(x.transpose(0, 1), y, target, preference.squeeze(1))


def sample_training_batch(batch_size, seq_len, device, epoch, max_features=None, max_objs=None):
    """Draw the context length before the per-batch problem dimensions."""
    max_features = DATA_CONFIG["max_features"] if max_features is None else max_features
    max_objs = DATA_CONFIG["max_objectives"] if max_objs is None else max_objs
    split = random.choices(range(1, seq_len), [1 / (seq_len - 1 - i) for i in range(seq_len - 1)])[0]
    num_features = random.randint(DATA_CONFIG["min_features"], max_features) if epoch > 0 else max_features
    num_objs = random.randint(DATA_CONFIG["min_objectives"], max_objs) if epoch > 0 else max_objs
    batch = get_batch(batch_size, seq_len, num_features, num_objs, device)
    batch.single_eval_pos = split
    return batch
