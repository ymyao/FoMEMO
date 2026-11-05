"""Native FoMEMO model loading and saving."""

import copy
from pathlib import Path

import torch

from .model import FoMEMOTransformer


def load_model(path, device):
    """Load the trusted FoMEMO checkpoint onto the selected device."""
    model = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(model, FoMEMOTransformer):
        raise TypeError("The checkpoint does not contain a FoMEMO inference model.")
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def save_model(model, path):
    """Save a CPU copy without changing the live model's parameters or mode."""
    saved = copy.deepcopy(model).to("cpu")
    # The objective-count distributions are held in a plain dictionary.
    for criterion in saved.criterion.values():
        criterion.to("cpu")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
