"""Minimal FoMEMO inference demo on a paper benchmark."""

from pathlib import Path

import torch

from fomemo import FoMEMOOptimizer
from benchmarks import get_benchmark


def main():
    torch.manual_seed(7)
    problem = get_benchmark("bc")
    train_x = torch.tensor(
        [
            [0.05, 0.20],
            [0.20, 0.80],
            [0.45, 0.45],
            [0.75, 0.15],
            [0.90, 0.70],
            [0.60, 0.95],
        ],
        dtype=torch.float64,
    )
    train_y = problem.evaluate(train_x)

    optimizer = FoMEMOOptimizer(
        model_path=Path(__file__).parent / "training_results" / "fomemo_model.pt",
        device="cpu",
        num_random_samples=16,
        num_restarts=2,
        num_grad_steps=2,
        num_preferences=8,
    )
    candidate = optimizer.suggest(
        train_x,
        train_y,
        batch_size=1,
        acquisition="ucb",
    )
    observation = problem.evaluate(candidate)
    print(f"benchmark: {problem.name}")
    print("candidate:")
    print(candidate.numpy())
    print("objectives:")
    print(observation)


if __name__ == "__main__":
    main()

# python demo.py