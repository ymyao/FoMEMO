# FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization

This repository provides the official implementation of the NeurIPS 2026 paper “FoMEMO: Towards Foundation Models for Expensive Multi-objective Optimization”.

## Installation

Run all commands from the project root.

```bash
conda env create -f environment.yml
conda activate fomemo
```

## Quick start

Create the `training_results/` directory and extract the pre-trained model from `fomemo_model.zip` into it to obtain `training_results/fomemo_model.pt`, then run the CPU demo:

```bash
python demo.py
```

## Training

Train a new model with a minimal CPU example:

```bash
python train.py --demo --device cpu --output-dir training_demo_results
```

For full training on one GPU:

```bash
python train.py
```

For distributed training on multiple GPUs, adjust the process count to your own setup:

```bash
torchrun --standalone --nproc_per_node=8 train.py
```

By default, training saves its outputs and final checkpoint to `training_results/`, with the model at `training_results/fomemo_model.pt`. Edit the default configuration in [fomemo/config.py](fomemo/config.py), or override the corresponding settings with command-line arguments. See `python train.py --help` for options.

## Optimization

Run repeated optimization trials on a benchmark:

```bash
python infer.py -p bc -q 1 -n 2 -t 3 -s hv2
```

This runs FoMEMO on `bc` problem with one candidate per iteration, two iterations per trial, and three independent trials, using `hv2` metric for evaluation.

By default, each trial runs all acquisition functions: `ei`, `ucb`, `uhvi`, and `ur2i`. Append `-a ucb` to run one method, or `-a uhvi ei` to run selected methods in that order. If metric computation is expensive, add `--no-verbose` to compute and report metrics only at the end of each trial. Use `-gpus` to specify the GPU index, the script will automatically use CPU if CUDA is unavailable.

Results are saved separately for each acquisition under `inference_results/q{batch_size}/{instance}/` as `res_{acquisition}_all.pkl`.

For your own observations, use the Python interface:

```python
from fomemo import FoMEMOOptimizer

optimizer = FoMEMOOptimizer(model_path="training_results/fomemo_model.pt")
candidates = optimizer.suggest(
    train_x, train_y, batch_size=1, acquisition="ucb"
)
```

`train_x` contains observed inputs stored as an `(n, d)` array or tensor, normalized to `[0, 1]`. `train_y` contains the corresponding raw objective values stored as an `(n, m)` array or tensor. All objectives are minimized. The returned candidates are CPU tensors with dimensions `(batch_size, d)` and values in `[0, 1]`.

## Benchmarks

Available benchmark problems in [benchmarks/](benchmarks/) include:

- Synthetic: `bc`, `ar`, `abc`, `abd`, `acd`, `bcd`.
- Engineering: `re21`, `re23`, `re24`, `re33`, `re35`, `re36`, `re37`, `re41`, `re42`, `re61`, `re91`.
- HPO: `lego`, `materials`, `mic`, `ship`.
- Additional studies: `wfg4_30d` (30d3m), `wfg4_40d` (40d3m), `weierstrass_takagi` (non-smooth).

## Evaluation metrics

Select a metric with `-s`:

| Argument | Description |
| --- | --- |
| `hv` | Hypervolume after ideal/nadir normalization, supported for engineering problems. |
| `hv2` | Hypervolume without additional objective normalization, supported for synthetic, HPO, and non-smooth problems. |
| `igd+` | IGD+ relative to a reference Pareto front, supported for `re91`, `wfg4_30d`, and `wfg4_40d`. |

Higher hypervolume and lower IGD+ indicate better performance.
