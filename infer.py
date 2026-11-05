"""Run FoMEMO optimization on benchmark problems."""

import argparse
import pickle
import time
from pathlib import Path

import torch
from botorch.utils.sampling import draw_sobol_samples

from fomemo import FoMEMOOptimizer
from fomemo.metric import _make_metric
from benchmarks import get_benchmark


ACQUISITIONS = ("ei", "ucb", "uhvi", "ur2i")


def main(TEST_INS, BATCH_SIZE, N_BATCH, N_TRIALS, METRIC,
         ACQ_NAMES=ACQUISITIONS, GPU_DEVICE=0, VERBOSE=True):

    # Prepare components
    if BATCH_SIZE < 1 or N_BATCH < 1 or N_TRIALS < 1:
        raise ValueError("BATCH_SIZE, N_BATCH and N_TRIALS must be positive.")
    if not ACQ_NAMES:
        raise ValueError("ACQ_NAMES must not be empty.")
    if any(name not in ACQUISITIONS for name in ACQ_NAMES):
        raise ValueError(f"ACQ_NAMES must contain only: {', '.join(ACQUISITIONS)}.")
    if len(set(ACQ_NAMES)) != len(ACQ_NAMES):
        raise ValueError("ACQ_NAMES must not contain duplicates.")
    device = torch.device(
        f"cuda:{GPU_DEVICE}" if torch.cuda.is_available() else "cpu"
    )
    problem = get_benchmark(TEST_INS)
    
    score, ideal, nadir = _make_metric(problem, METRIC, device)
    optimizer = FoMEMOOptimizer(
        model_path=Path(__file__).parent / "training_results" / "fomemo_model.pt", device=device
    )
    results = {acquisition: [] for acquisition in ACQ_NAMES}
    folder = Path("inference_results") / f"q{BATCH_SIZE}" / TEST_INS

    print("-------------------FoMEMO inference-------------------", flush=True)
    print(f"test instance: {TEST_INS} ({problem.name})")
    print(f"variables: {problem.n_dim}, objectives: {problem.n_obj}")
    print(f"ideal point: {ideal.cpu().numpy()}")
    print(f"nadir point: {nadir.cpu().numpy()}")
    print(f"batch size: {BATCH_SIZE}, iterations: {N_BATCH}, trials: {N_TRIALS}")
    print(f"metric: {METRIC}, device: {device}, verbose: {VERBOSE}")

    # Start optimization
    for trial in range(1, N_TRIALS + 1):
        print(f"\n[Trial {trial}/{N_TRIALS}]", flush=True)
        n_initial = 2 * (problem.n_dim + 1)
        initial_x = draw_sobol_samples(
            bounds=torch.tensor([[0.0] * problem.n_dim, [1.0] * problem.n_dim], device=device), 
            n=n_initial, q=1).squeeze(1).to(dtype=torch.float64)
        initial_y = torch.as_tensor(
            problem.evaluate(initial_x), dtype=torch.float64, device=device
        )
        initial_hvs = [score(initial_y)] if VERBOSE else []
        histories = {
            acquisition: {
                "train_x": initial_x.clone(),
                "train_y": initial_y.clone(),
                "hvs": initial_hvs.copy(),
                "time": [],
            }
            for acquisition in ACQ_NAMES
        }
        print(f"Generated {n_initial} initial samples", flush=True)

        for iteration in range(1, N_BATCH + 1):
            for acquisition in ACQ_NAMES:
                history = histories[acquisition]
                search_start = time.monotonic()
                candidate = optimizer.suggest(
                    history["train_x"], history["train_y"],
                    batch_size=BATCH_SIZE, acquisition=acquisition,
                )
                history["time"].append(time.monotonic() - search_start)
                observation = torch.as_tensor(
                    problem.evaluate(candidate), dtype=torch.float64, device=device
                )
                history["train_x"] = torch.cat([
                    history["train_x"], candidate.to(device=device, dtype=torch.float64)
                ])
                history["train_y"] = torch.cat([history["train_y"], observation])

            if VERBOSE:
                for history in histories.values():
                    history["hvs"].append(score(history["train_y"]))
            details = []
            if VERBOSE:
                values = ", ".join(f"{histories[acquisition]['hvs'][-1]:.3g}" for acquisition in ACQ_NAMES)
                details.append(f"{METRIC} ({', '.join(ACQ_NAMES)}) = ({values})")
            query_times = ", ".join(f"{histories[acquisition]['time'][-1]:.2f}" for acquisition in ACQ_NAMES)
            details.append(f"query time ({', '.join(ACQ_NAMES)}) = ({query_times})s")
            print(
                f"Batch {iteration}/{N_BATCH}: "
                + ", ".join(details),
                flush=True,
            )

        if not VERBOSE:
            for history in histories.values():
                history["hvs"].append(score(history["train_y"]))
            values = ", ".join(f"{histories[acquisition]['hvs'][-1]:.3g}" for acquisition in ACQ_NAMES)
            print(
                f"Trial {trial} final {METRIC} ({', '.join(ACQ_NAMES)}) = ({values})",
                flush=True,
            )

        # Save results
        folder.mkdir(parents=True, exist_ok=True)
        for acquisition, history in histories.items():
            results[acquisition].append({
                "train_x": history["train_x"].detach().cpu(),
                "train_y": history["train_y"].detach().cpu(),
                "hvs": history["hvs"],
                "time": history["time"],
            })
            with (folder / f"res_{acquisition}_all.pkl").open("wb") as handle:
                pickle.dump(results[acquisition], handle)
        print(f"Trial {trial} saved to {folder}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", type=str, required=True, help="Benchmark problem name")
    parser.add_argument("-q", type=int, required=True, help="Number of candidates per iteration")
    parser.add_argument("-n", type=int, required=True, help="Number of optimization iterations per trial")
    parser.add_argument("-t", type=int, required=True, help="Number of independent trials")
    parser.add_argument("-s", type=str, required=True, help="Score metric: hv, hv2, or igd+")
    parser.add_argument(
        "-a", nargs="+", choices=ACQUISITIONS,
        default=ACQUISITIONS, help="Acquisition functions to run",
    )
    parser.add_argument("-gpus", type=int, default=0, help="GPU device index (default: 0)")
    parser.add_argument("--no-verbose", dest="verbose", action="store_false", help="Report metrics only at trial end")
    args = parser.parse_args()

    TEST_INS = args.p
    BATCH_SIZE = args.q
    N_BATCH = args.n
    N_TRIALS = args.t
    METRIC = args.s
    ACQ_NAMES = args.a
    GPU_DEVICE = args.gpus
    VERBOSE = args.verbose

    main(TEST_INS, BATCH_SIZE, N_BATCH, N_TRIALS, METRIC, ACQ_NAMES, GPU_DEVICE, VERBOSE)



# CUDA_VISIBLE_DEVICES=-1 python infer.py -p bc -q 1 -n 2 -t 3 -s hv2

# CUDA_VISIBLE_DEVICES=0 python infer.py -p re23 -q 1 -n 2 -t 3 -s hv

# CUDA_VISIBLE_DEVICES=0 python infer.py -p mat -q 1 -n 2 -t 3 -s hv2

# CUDA_VISIBLE_DEVICES=0 python infer.py -p wfg4_30d -q 1 -n 2 -t 3 -s hv