"""Experiment 5.1: Ambient Gaussian smoothing on the UNIT SPHERE S^m.

This experiment investigates how the ambient dimension D affects the squared
Wasserstein error W_2^2(k_sigma * mu_n, mu) and its optimal smoothing bandwidth.
Here mu is the uniform probability measure on S^m, mu_n is the empirical measure
of n independent samples from mu, and k_sigma is the Gaussian kernel with
covariance sigma^2 I_D. The sphere is embedded in R^D by zero-padding its
coordinates in R^(m+1), with D >= m + 1.

For fixed n and m, the experiment examines the following predicted trends:
1. The minimum error over bandwidths increases and the optimal bandwidth
   decreases as D increases.
2. The optimal bandwidth exhibits approximate inverse-dimension scaling,
   sigma_amb proportional to 1/D.
These trends are assessed numerically using a finite bandwidth grid.

Default configuration:
    m = 3 (S^3; use --m 4 for S^4), n = 50, N_eval = 5000, and five independent
    trials, with sigma_j = 0.6 * (j / 80)^2 for j = 0, ..., 80.
    L1 = {6, 8, 10, 12, 14} is used to compare mean error curves across D.
    L2 = {10, 20, 40, 80, 160, 320} is used to examine bandwidth scaling with D.
Dimensions shared by L1 and L2 are computed only once per trial. These defaults
can be changed through command-line arguments.

Sampling and discretization:
    For each trial, draw n independent points X_i from mu to form mu_n and an
    independent target sample of N_eval points Z_j from mu. Reuse these samples
    across all ambient dimensions and bandwidths within that trial.

    For each (trial, D), independently draw N_eval indices J_j uniformly from
    {1, ..., n} and N_eval standard Gaussian vectors eta_j in R^D. Reuse these
    indices and Gaussian vectors across all bandwidths, forming
        Y_j(sigma) = embed_D(X_{J_j}) + sigma * eta_j.
    The equally weighted points Y_j(sigma) approximate k_sigma * mu_n, and the
    equally weighted points embed_D(Z_j) approximate mu.

OT computation and reference costs:
    Use POT's ot.emd2() with uniform weights and squared Euclidean costs to
    compute OT between the two N_eval-point clouds. Also compute unsmoothed
    reference costs directly between the n empirical points and the N_eval
    target points, using both squared Euclidean and squared spherical-geodesic
    distances on the unit sphere.

    At sigma = 0, the smoothing curve uses N_eval resampled empirical points.
    Its value can therefore differ from the direct n-point Euclidean reference
    because the resampled empirical weights fluctuate.

Numerical optimal bandwidth:
    After the experiment, plot_results.py averages the squared OT costs across
    trials at each (D, sigma) and defines sigma_amb as the grid bandwidth that
    minimizes this mean curve. The search includes sigma = 0; ties are resolved
    by choosing the smallest bandwidth. This is a finite-grid estimate, rather
    than a continuous optimization over strictly positive bandwidths.

Saving and plotting:
    Keep this script and plot_results.py in the same folder. Running this script
    creates a timestamped run subfolder beside it (or under --output-dir), saves
    the configuration, measurements, and random draws, copies plot_results.py,
    and creates a Windows plot_results.cmd launcher. Plotting is a separate step:
    after the computation finishes, run plot_results.py in the run subfolder,
    or double-click plot_results.cmd on Windows.

Code developed with assistance from Codex GPT-6 Astra.
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
        raise SystemExit(subprocess.call([str(PROJECT_PYTHON), "-B", str(Path(__file__).resolve()), *sys.argv[1:]]))

import numpy as np
import ot


# store variables in a JSON file for reproducibility, and store the measurements in a CSV file
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=3, help="intrinsic dimension of the unit sphere S^m") # the intrinsic dimension of the unit sphere
    parser.add_argument("--n", type=int, default=50) # the number of iid samples for mu_n
    parser.add_argument("--n-eval", type=int, default=5000) # N_eval-point cloud for discretization of marginal measures
    parser.add_argument("--num-trials", type=int, default=5) # the number of independent trials to run
    parser.add_argument("--suite-1-dimensions", type=int, nargs="+",
                        default=[6, 8, 10, 12, 14]) # dimensions for illustrating W_2^2 curves
    parser.add_argument("--suite-2-dimensions", type=int, nargs="+",
                        default=[10, 20, 40, 80, 160, 320]) # dimensions for illustrating the scaling of sigma_amb with D
    parser.add_argument("--sigma-max", type=float, default=0.6) # the maximum bandwidth for the Gaussian smoothing kernel
    parser.add_argument("--seed", type=int, default=20260921) # the random seed for reproducibility
    parser.add_argument("--output-dir", type=Path, default=HERE) 
    args = parser.parse_args()

    # Validate the arguments to ensure they are positive and meet the requirements for the experiment
    if min(args.m, args.n, args.n_eval) < 1 or args.num_trials < 2:
        parser.error("m, n and n-eval must be positive; num-trials must be at least 2")
    for name, suite in (("suite 1", args.suite_1_dimensions),
                        ("suite 2", args.suite_2_dimensions)):
        if not suite or len(set(suite)) != len(suite):
            parser.error(f"{name} dimensions must be nonempty and distinct")
        if min(suite) < args.m + 1:
            parser.error(f"all {name} dimensions must be at least m + 1 = {args.m + 1}")
    # Preserve suite order and compute dimensions shared by both suites only once.
    args.dimensions = list(dict.fromkeys(
        args.suite_1_dimensions + args.suite_2_dimensions))
    if not np.isfinite(args.sigma_max) or args.sigma_max <= 0 or args.seed < 0:
        parser.error("sigma-max must be finite and positive; seed must be nonnegative")
    return args

# sample a uniform point on the unit sphere S^m in R^(m+1) by normalizing a standard Gaussian vector
def sample_sphere(size, m, rng):
    """Uniform points on S^m in R^(m+1)."""
    points = rng.normal(size=(size, m + 1))
    return points / np.linalg.norm(points, axis=1, keepdims=True)

# compute the squared 2-Wasserstein distance between two point clouds using the squared Euclidean cost and POT's
# emd2 solver, which is a wrapper around the network simplex algorithm.
def w2_squared(source, target):
    """Same squared-cost construction and public POT call as sphere 5.1.1."""
    source_norms = np.einsum("ij,ij->i", source, source) # compute the squared norms of the source points
    target_norms = np.einsum("ij,ij->i", target, target) # compute the squared norms of the target points
    inner = np.einsum("ik,jk->ij", source, target, optimize=False) # compute the inner products between source and target points
    cost = source_norms[:, None] + target_norms[None, :] - 2.0 * inner
    np.maximum(cost, 0.0, out=cost)
    # compute W_2^2(source, target) by ot.emd2() with uniform weights and the computed cost 
    value = ot.emd2(
        np.full(len(source), 1.0 / len(source)),
        np.full(len(target), 1.0 / len(target)),
        cost,
        numItermax=1_000_000,
    )
    return float(max(value, 0.0))

# Compute unsmoothed OT on S^m using squared great-circle distance.
def geodesic_w2_squared(source, target):
    """Squared spherical-geodesic OT cost between unit-sphere point clouds."""
    inner = np.einsum("ik,jk->ij", source, target, optimize=False)
    np.clip(inner, -1.0, 1.0, out=inner)
    cost = np.arccos(inner) ** 2 # compute geodsic disctance on the unit sphere S^m
    # compute W_2^2(source, target) by ot.emd2() with uniform weights and the computed cost
    value = ot.emd2(
        np.full(len(source), 1.0 / len(source)),
        np.full(len(target), 1.0 / len(target)),
        cost,
        numItermax=1_000_000,
    )
    return float(max(value, 0.0))

# the function to run the experiment, which streams measurements to a CSV file and saves all draws for reproducibility
def run_experiment(args, directory):
    """Stream measurements to CSV and retain all draws for reproducibility."""
    # The sigma grid is quadratic, so that the smallest sigma is 0 and the largest is args.sigma_max
    sigmas = args.sigma_max * (np.arange(81) / 80) ** 2

    # fields represent the columns of the CSV file, which are the trial number, ambient dimension, grid index, sigma
    # the smoothed and unsmoothed squared 2-Wasserstein distances, and the time taken to compute the cost and OT
    # distance
    fields = ["trial", "ambient_dimension", "grid_index", "sigma",
              "smoothed_euclidean_w2_squared", "unsmoothed_euclidean_w2_squared",
              "unsmoothed_geodesic_w2_squared",
              "cost_and_ot_seconds"]

    # Write the CSV header and then stream the measurements for each trial, dimension, and sigma to the file
    with (directory / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        # Run the experiment for args.num_trials, with the empirical and target samples shared across D and sigma 
        for trial in range(1, args.num_trials + 1):

            # The empirical and target samples are shared across D and sigma.
            rng = np.random.default_rng(np.random.SeedSequence([args.seed, trial, 0]))
            empirical = sample_sphere(args.n, args.m, rng)
            target = sample_sphere(args.n_eval, args.m, rng)
            # Compute the direct unsmoothed baselines once per trial.
            euclidean_baseline = w2_squared(empirical, target) # under Euclidean metric
            geodesic_baseline = geodesic_w2_squared(empirical, target) # under Spherical geodesic metric

            # Save the empirical and target samples for reproducibility.
            np.savez(directory / f"samples_trial_{trial}.npz",
                     empirical=empirical, target=target)
            print(
                f"Trial {trial}/{args.num_trials}: "
                f"Euclidean baseline={euclidean_baseline:.10g}, "
                f"geodesic baseline={geodesic_baseline:.10g}",
                flush=True,
            )

            # For each ambient dimension, embed the empirical and target samples in R^D by zero padding
            for dimension in args.dimensions:
                # np.pad is used to embed the target samples in R^D by adding zeros to the additional dimensions
                embedded_target = np.pad(target, ((0, 0), (0, dimension - (args.m + 1))))
                
                # Draw one mixture discretization for this (trial, D) and reuse it
                # across all bandwidths so only the noise scale changes with sigma.
                index_rng = np.random.default_rng(
                    np.random.SeedSequence([args.seed, trial, dimension, 2]))
                indices = index_rng.integers(0, args.n, size=args.n_eval)
                np.save(directory / f"indices_trial_{trial}_D{dimension}.npy", indices)
                centers = np.pad(
                    empirical[indices],
                    ((0, 0), (0, dimension - (args.m + 1))),
                )

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

                    writer.writerow(dict(zip(fields, [trial, dimension, j, float(sigma),
                                                     value, euclidean_baseline,
                                                     geodesic_baseline, elapsed])))
                    handle.flush()
                    print(
                        f"  D={dimension}, sigma {j + 1}/81: {sigma:.6g}, "
                        f"W2^2={value:.10g} ({elapsed:.1f} sec)",
                        flush=True,
                    )


def main():
    args = parse_args() # parse command-line arguments for experiment
    started = datetime.now().astimezone() # record the start time of the experiment
    dimension_label = f"{min(args.dimensions)}-{max(args.dimensions)}x{len(args.dimensions)}"
    # Create a unique directory for the experiment based on parameters and timestamp
    directory = args.output_dir.resolve() / (
        f"S{args.m}_D{dimension_label}_n{args.n}_N{args.n_eval}_T{args.num_trials}_G81_"
        f"{started.strftime('%Y%m%d-%H%M%S-%f')}")
    directory.mkdir(parents=True, exist_ok=False)

    # Save the experiment configuration to a JSON file for reproducibility, including parameters, 
    # random seeds, and versions of libraries used
    config = {
        **vars(args), "output_dir": str(args.output_dir.resolve()),
        "manifold": f"S^{args.m}", "intrinsic_dimension": args.m,
        "sigma_values": (args.sigma_max * (np.arange(81) / 80) ** 2).tolist(),
        "num_sigmas": 81, "sigma_min": 0.0,
        "sigma_grid": "quadratic", "solver": "ot.emd2", "numItermax": 1_000_000,
        "smoothed_cost": "squared Euclidean",
        "unsmoothed_reference_costs": ["squared Euclidean", "squared spherical geodesic"],
        "weights": "uniform on each discrete marginal",
        "randomness": "Empirical and target samples shared across D and sigma per trial; fresh independent uniform indices and Gaussian noise for every (trial, D), reused across all sigma values within that (trial, D).",
        "seed_policy": "Samples: SeedSequence([seed, trial, 0]); noise: SeedSequence([seed, trial, D, 1]); indices: SeedSequence([seed, trial, D, 2]); trials start at 1.",
        "zero_sigma": f"{args.n_eval} resampled empirical points",
        "standard_deviation": "sample SD, ddof=1", "numpy_version": np.__version__,
        "pot_version": ot.__version__, "python": sys.executable,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_started_at": started.isoformat(), "status": "running",
    }
    config_path = directory / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(HERE / "plot_results.py", directory / "plot_results.py")
    (directory / "plot_results.cmd").write_text(
        f'@echo off\n"{sys.executable}" "%~dp0plot_results.py"\npause\n', encoding="utf-8")
    print(f"Saving to {directory}", flush=True)
    run_experiment(args, directory)
    config.update(status="complete", run_ended_at=datetime.now().astimezone().isoformat())
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Finished. Run {directory / 'plot_results.cmd'} to create the plots.")


if __name__ == "__main__":
    main()
