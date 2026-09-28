"""Exact-OT smoothing experiment on a product of spheres.

Within each trial, the empirical and target samples are fixed across D and
sigma. For each D, fresh mixture indices and Gaussian noise are reused across
the 81 bandwidths. Run plot_results.cmd in the saved run folder afterward.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).resolve().parent
PROJECT_PYTHON = HERE.parents[1] / ".venv-experiment-5-1-1" / "Scripts" / "python.exe"
if __name__ == "__main__" and PROJECT_PYTHON.exists():
    if Path(sys.executable).resolve() != PROJECT_PYTHON.resolve():
        raise SystemExit(subprocess.call(
            [str(PROJECT_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]
        ))

import numpy as np
import ot

# store variables in a JSON file for reproducibility, and store the measurements in a CSV file
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1", type=int, default=1) # set the intrinsic dimension of the first sphere
    parser.add_argument("--m2", type=int, default=2) # set the intrinsic dimension of the second sphere
    parser.add_argument("--n", type=int, default=50) # set the number of iid samples for mu_n
    parser.add_argument("--n-eval", type=int, default=500) # set N_eval samples for discretization of the marginal distributions 
    parser.add_argument("--num-trials", type=int, default=5) # set the number of trials to run
    parser.add_argument("--suite-1-dimensions", type=int, nargs="+",
                        default=[6, 8, 10, 12, 14]) # set the dimensions for the first suite of experiments: for illustrating W_2^2 curves 
    parser.add_argument("--suite-2-dimensions", type=int, nargs="+",
                        default=[10, 20, 40, 80, 160, 320]) # set the dimensions for the second suite of experiments: for illustrating the scaling of sigma_amb on the ambient dimension
    parser.add_argument("--sigma-max", type=float, default=0.6) # set the maximum bandwidth 
    parser.add_argument("--seed", type=int, default=20260921) 
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    # Validate the arguments to ensure they are positive and meet the requirements for the experiment
    if min(args.m1, args.m2, args.n, args.n_eval) < 1 or args.num_trials < 2:
        parser.error("m1, m2, n and n-eval must be positive; num-trials must be at least 2")
    coordinate_dimension = args.m1 + args.m2 + 2
    for name, suite in (("suite 1", args.suite_1_dimensions),
                        ("suite 2", args.suite_2_dimensions)):
        if not suite or len(set(suite)) != len(suite):
            parser.error(f"{name} dimensions must be nonempty and distinct")
        if min(suite) < coordinate_dimension:
            parser.error(f"all {name} dimensions must be at least {coordinate_dimension}")
    if not np.isfinite(args.sigma_max) or args.sigma_max <= 0 or args.seed < 0:
        parser.error("sigma-max must be finite and positive; seed must be nonnegative")

    # Preserve suite order and compute dimensions shared by both suites only once.    
    args.dimensions = list(dict.fromkeys(
        args.suite_1_dimensions + args.suite_2_dimensions
    ))
    return args


def sample_product_spheres(size, m1, m2, rng):
    """Sample uniformly from S^m1 x S^m2 in R^(m1+m2+2)."""
    parts = []
    # sample from each sphere and normalize to lie on the sphere
    for coordinate_dimension in (m1 + 1, m2 + 1):
        points = rng.normal(size=(size, coordinate_dimension))
        parts.append(points / np.linalg.norm(points, axis=1, keepdims=True))
    return np.concatenate(parts, axis=1) # return concatenated points in R^(m1+m2+2) as iid samples from the product of spheres

# compute W_2^2 using the ot.emd2 function with the given cost matrix
def exact_w2_squared(source, target, cost):
    value = ot.emd2(
        np.full(len(source), 1.0 / len(source)),
        np.full(len(target), 1.0 / len(target)),
        cost,
        numItermax=1_000_000,
    )
    return float(max(value, 0.0))

# compute W_2^2 for the squared Euclidean cost
def w2_squared(source, target):
    """Exact W2 squared for the squared Euclidean ground cost."""
    source_norms = np.einsum("ij,ij->i", source, source)
    target_norms = np.einsum("ij,ij->i", target, target)
    inner = np.einsum("ik,jk->ij", source, target, optimize=False)
    cost = source_norms[:, None] + target_norms[None, :] - 2.0 * inner
    np.maximum(cost, 0.0, out=cost)
    return exact_w2_squared(source, target, cost)

# compute W_2^2 for the product-sphere geodesic cost (for unsmoothed W_2^2(mu_n, mu))
def geodesic_w2_squared(source, target, split):
    """Exact W2 squared for the product-sphere geodesic ground cost."""
    first_source, second_source = source[:, :split], source[:, split:]
    first_target, second_target = target[:, :split], target[:, split:]
    first_inner = np.einsum("ik,jk->ij", first_source, first_target, optimize=False)
    second_inner = np.einsum("ik,jk->ij", second_source, second_target, optimize=False)
    first_arc = np.arccos(np.clip(first_inner, -1.0, 1.0))
    second_arc = np.arccos(np.clip(second_inner, -1.0, 1.0))
    cost = first_arc ** 2 + second_arc ** 2 # compute the squared geodesic distance on the product of spheres
    return exact_w2_squared(source, target, cost)



# the function to run the experiment, which streams measurements to a CSV file and saves all draws for reproducibility
def run_experiment(args, directory):
    # The sigma grid is quadratic, so that the smallest sigma is 0 and the largest is args.sigma_max
    sigmas = args.sigma_max * (np.arange(81) / 80) ** 2
    coordinate_dimension = args.m1 + args.m2 + 2

    # Stream the measurements to a CSV file and save all draws for reproducibility
    fields = [
        "trial", "ambient_dimension", "grid_index", "sigma",
        "smoothed_euclidean_w2_squared", "unsmoothed_euclidean_w2_squared",
        "unsmoothed_geodesic_w2_squared", "cost_and_ot_seconds",
    ]

    # write the measurements to a CSV file and save all draws for reproducibility
    with (directory / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        # Run the experiment for args.num_trials, with the empirical and target samples shared across D and sigma 
        for trial in range(1, args.num_trials + 1):

            # The empirical and target samples are shared across D and sigma.
            sample_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, trial, 0]))
            empirical = sample_product_spheres(args.n, args.m1, args.m2, sample_rng)
            target = sample_product_spheres(args.n_eval, args.m1, args.m2, sample_rng)

            # Compute the direct unsmoothed baselines once per trial.
            euclidean_baseline = w2_squared(empirical, target)
            geodesic_baseline = geodesic_w2_squared(
                empirical, target, split=args.m1 + 1)

            # Save the empirical and target samples for reproducibility.
            np.savez(directory / f"samples_trial_{trial}.npz",
                     empirical=empirical, target=target)
            print(f"Trial {trial}/{args.num_trials}: "
                  f"Euclidean baseline={euclidean_baseline:.10g}, "
                  f"geodesic baseline={geodesic_baseline:.10g}", flush=True)

            # For each ambient dimension, embed the empirical and target samples in R^D by zero padding
            for dimension in args.dimensions:
                # np.pad is used to embed the target samples in R^D by adding zeros to the additional dimensions
                padding = ((0, 0), (0, dimension - coordinate_dimension))
                embedded_target = np.pad(target, padding)
                
                # Draw one mixture discretization for this (trial, D) and reuse it
                # across all bandwidths so only the noise scale changes with sigma.
                index_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, trial, dimension, 2]))
                indices = index_rng.integers(0, args.n, size=args.n_eval)
                np.save(directory / f"indices_trial_{trial}_D{dimension}.npy", indices)
                centers = np.pad(empirical[indices], padding)

                noise_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, trial, dimension, 1]))
                noise = noise_rng.normal(size=(args.n_eval, dimension))
                np.save(directory / f"noise_trial_{trial}_D{dimension}.npy", noise)

                for j, sigma in enumerate(sigmas):

                    # The same centers and Gaussian vectors are scaled across sigma.
                    source = centers + sigma * noise

                    started = perf_counter() # record the time taken to compute the cost and OT distance

                    # compute W_2^2(k_sigma * mu_n , mu) using ot.emd2()
                    value = w2_squared(source, embedded_target) 
                    elapsed = perf_counter() - started
                    writer.writerow(dict(zip(fields, [
                        trial, dimension, j, float(sigma), value,
                        euclidean_baseline, geodesic_baseline, elapsed,
                    ])))
                    handle.flush()
                    print(f"  D={dimension}, sigma {j + 1}/81: {sigma:.6g}, "
                          f"W2^2={value:.10g} ({elapsed:.1f} sec)", flush=True)


def main():
    args = parse_args() # parse command-line arguments for experiment
    started = datetime.now().astimezone() # record the start time of the experiment
    dimension_label = f"{min(args.dimensions)}-{max(args.dimensions)}x{len(args.dimensions)}"

    # Create a unique directory for the experiment based on parameters and timestamp
    directory = args.output_dir.resolve() / (
        f"S{args.m1}xS{args.m2}_D{dimension_label}_n{args.n}_N{args.n_eval}_"
        f"T{args.num_trials}_G81_{started.strftime('%Y%m%d-%H%M%S-%f')}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    
    # Save the experiment configuration to a JSON file for reproducibility, including parameters, 
    # random seeds, and versions of libraries used
    config = {
        **vars(args), "output_dir": str(args.output_dir.resolve()),
        "manifold": f"S^{args.m1} x S^{args.m2}",
        "intrinsic_dimension": args.m1 + args.m2,
        "coordinate_dimension": args.m1 + args.m2 + 2,
        "sigma_values": (args.sigma_max * (np.arange(81) / 80) ** 2).tolist(),
        "num_sigmas": 81, "sigma_min": 0.0, "sigma_grid": "quadratic",
        "solver": "ot.emd2", "numItermax": 1_000_000,
        "smoothed_cost": "squared Euclidean",
        "unsmoothed_reference_costs": [
            "squared Euclidean", "squared product-sphere geodesic",
        ],
        "weights": "uniform on each discrete marginal",
        "randomness": (
            "Empirical and target samples shared across D and sigma per trial; "
            "fresh independent uniform indices and Gaussian noise for every "
            "(trial, D), reused across sigma."
        ),
        "seed_policy": (
            "Samples: SeedSequence([seed, trial, 0]); "
            "noise: SeedSequence([seed, trial, D, 1]); "
            "indices: SeedSequence([seed, trial, D, 2]); trials start at 1."
        ),
        "zero_sigma": f"{args.n_eval} resampled empirical points",
        "standard_deviation": "sample SD, ddof=1",
        "numpy_version": np.__version__, "pot_version": ot.__version__,
        "python": sys.executable,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_started_at": started.isoformat(), "status": "running",
    }
    config_path = directory / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(HERE / "plot_results.py", directory / "plot_results.py")
    (directory / "plot_results.cmd").write_text(
        f'@echo off\n"{sys.executable}" "%~dp0plot_results.py"\npause\n',
        encoding="utf-8",
    )
    print(f"Saving to {directory}", flush=True)
    run_experiment(args, directory)
    config.update(status="complete", run_ended_at=datetime.now().astimezone().isoformat())
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Finished. Run {directory / 'plot_results.cmd'} to create the plots.")


if __name__ == "__main__":
    main()
