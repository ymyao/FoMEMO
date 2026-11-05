"""Calibration data and model construction for FoMEMO training."""

import math
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import CALIBRATION_CONFIG, DATA_CONFIG, DEMO_CONFIG, MODEL_CONFIG, TRAINING_CONFIG
from .distributions import FullSupportBarDistribution
from .model import FoMEMOTransformer, Linear, Normalize, VariableNumFeaturesEncoder
from .prior import get_batch


@contextmanager
def isolated_random_state(device):
    """Keep calibration sampling independent of the training random stream."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def calibration_settings(demo=False):
    if demo:
        return tuple(DEMO_CONFIG[f"calibration_{key}"] for key in ("dimensions", "trials", "batch_size", "seq_len"))
    return tuple(CALIBRATION_CONFIG[key] for key in ("dimensions", "trials", "batch_size", "seq_len"))


def validate_config(args):
    """Check effective budgets and compatible model, data and calibration sizes."""
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}.")

    for key, value in MODEL_CONFIG.items():
        integer(value, f"MODEL_CONFIG[{key!r}]", 2 if key == "output_logits_size" else 1)
    if MODEL_CONFIG["embedding_size"] % MODEL_CONFIG["attention_heads"]:
        raise ValueError("Embedding size must be divisible by attention heads.")
    for name in ("epochs", "steps_per_epoch", "batch_size", "seq_len", "warmup_epochs"):
        integer(getattr(args, name), name, 0 if name == "warmup_epochs" else (2 if name == "seq_len" else 1))
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    integer(TRAINING_CONFIG["checkpoint_interval"], "checkpoint_interval")
    for low, high, capacity, minimum in (
        ("min_features", "max_features", "max_encoder_features", 1),
        ("min_objectives", "max_objectives", "max_encoder_objectives", 2),
    ):
        integer(DATA_CONFIG[low], low, minimum)
        integer(DATA_CONFIG[high], high, minimum)
        if not DATA_CONFIG[low] <= DATA_CONFIG[high] <= MODEL_CONFIG[capacity]:
            raise ValueError(f"Require {low} <= {high} <= {capacity}.")
    for key in ("gp_output_scale", "gp_noise_variance", "lengthscale_concentration", "lengthscale_rate", "sigmoid_scale", "preference_concentration"):
        if not math.isfinite(DATA_CONFIG[key]) or DATA_CONFIG[key] <= 0:
            raise ValueError(f"{key} must be finite and positive.")
    if not math.isfinite(DATA_CONFIG["gp_mean"]):
        raise ValueError("GP mean must be finite.")
    if not 0 <= DATA_CONFIG["gp_noise_lower_bound"] < DATA_CONFIG["gp_noise_variance"]:
        raise ValueError("GP noise variance must exceed its nonnegative lower bound.")
    if not callable(DATA_CONFIG["gp_kernel"]):
        raise ValueError("GP kernel must be a callable kernel class.")
    flags = DATA_CONFIG["gp_fast_computations"]
    if len(flags) != 3 or any(not isinstance(flag, bool) for flag in flags):
        raise ValueError("GP fast computations requires three boolean flags.")
    dimensions, trials, batch_size, seq_len = calibration_settings(args.demo)
    if not dimensions:
        raise ValueError("Calibration dimensions must not be empty.")
    for dimension in dimensions:
        integer(dimension, "calibration dimension")
        if dimension > MODEL_CONFIG["max_encoder_features"]:
            raise ValueError("Calibration dimensions exceed the feature encoder capacity.")
    for name, value in (("calibration trials", trials), ("calibration batch size", batch_size), ("calibration sequence length", seq_len)):
        integer(value, name)
    if len(dimensions) * trials * batch_size * seq_len < MODEL_CONFIG["output_logits_size"]:
        raise ValueError("Calibration requires at least one sample per output bucket.")
    ood = CALIBRATION_CONFIG["ood_objectives"]
    if len(set(ood)) != len(ood):
        raise ValueError("OOD objective counts must be unique.")
    for count in ood:
        integer(count, "OOD objective count", 2)
        if count > MODEL_CONFIG["max_encoder_objectives"] or DATA_CONFIG["min_objectives"] <= count <= DATA_CONFIG["max_objectives"]:
            raise ValueError("OOD objective counts must fit the encoder and not overlap training.")


def get_bucket_limits(ys, num_outputs=None):
    """Estimate equal-mass buckets using sorted adjacent-sample midpoints."""
    num_outputs = MODEL_CONFIG["output_logits_size"] if num_outputs is None else num_outputs
    ys = ys.flatten()
    ys = ys[~torch.isnan(ys)]
    if len(ys) % num_outputs:
        ys = ys[:-(len(ys) % num_outputs)]
    if len(ys) < num_outputs:
        raise ValueError("Calibration requires at least one sample per bucket.")
    low, high = ys.min(), ys.max()
    per_bucket = len(ys) // num_outputs
    ordered, _ = ys.sort(0)
    middle = (ordered[per_bucket - 1::per_bucket][:-1] + ordered[per_bucket::per_bucket]) / 2
    return torch.cat([low.unsqueeze(0), middle, high.unsqueeze(0)], 0)


def generate_ys(directory, objectives, device, demo=False):
    """Generate scalar targets with the full calibration budget or the demo budget."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dimensions, trials, batch_size, seq_len = calibration_settings(demo)
    for num_objs in objectives:
        targets = []
        for num_features in dimensions:
            print(f"  objectives={num_objs}, variables={num_features}, batches={trials}", flush=True)
            for _ in range(trials):
                batch = get_batch(batch_size, seq_len, num_features, num_objs, device)
                targets.append(batch.target_y.cpu().flatten())
        torch.save(torch.cat(targets, 0), directory / f"ys_objs_{num_objs}.pt")


def make_criteria(directory, objectives):
    criteria = {}
    for num_objs in objectives:
        ys = torch.load(Path(directory) / f"ys_objs_{num_objs}.pt", map_location="cpu", weights_only=True)
        criteria[num_objs] = FullSupportBarDistribution(get_bucket_limits(ys))
    return criteria


def build_model(criteria):
    if any(criterion.num_bars != MODEL_CONFIG["output_logits_size"] for criterion in criteria.values()):
        raise ValueError("Criterion bucket counts must match the model output size.")

    def encoder(num_features):
        return nn.Sequential(
            Normalize(0.5, math.sqrt(1 / 12)),
            VariableNumFeaturesEncoder(Linear(num_features, MODEL_CONFIG["embedding_size"]), num_features),
        )

    # Preserve the initialization order: preference, input, objectives, Transformer.
    style_encoder = encoder(MODEL_CONFIG["max_encoder_objectives"])
    x_encoder = encoder(MODEL_CONFIG["max_encoder_features"])
    y_encoder = encoder(MODEL_CONFIG["max_encoder_objectives"])
    model = FoMEMOTransformer(x_encoder, y_encoder, style_encoder)
    model.criterion = criteria
    return model


def attach_ood_criteria(model, directory):
    """Add objective-count distributions without updating the trained network."""
    expected = set(range(DATA_CONFIG["min_objectives"], DATA_CONFIG["max_objectives"] + 1))
    if set(model.criterion) != expected:
        raise ValueError(f"The trained model must contain distributions for objectives {sorted(expected)}.")
    model.criterion.update(make_criteria(directory, CALIBRATION_CONFIG["ood_objectives"]))
    return model
