"""Experiment 5.1.2 on S^m1 x S^m2, with mixture draws reused across sigma.

Run this file to compute and save a new experiment. Run plot_results.cmd in
the resulting run folder separately to plot its saved distances.
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
    parser.add_argument("--m1", type=int, default=1, help="intrinsic dimension of the first unit sphere") # intrinsic dimension of the first unit sphere
    parser.add_argument("--m2", type=int, default=2, help="intrinsic dimension of the second unit sphere") # intrinsic dimension of the second unit sphere
    parser.add_argument("--D", type=int, default=15, help="fixed ambient dimension") # fixed ambient dimension
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[5, 10, 20, 40, 80]) # sample sizes for illustrating W_2^2 curves and the scaling of sigma_amb on n
    parser.add_argument("--n-eval", type=int, default=5000) # N_eval-point cloud for discretization of marginal measures
    parser.add_argument("--num-trials", type=int, default=5) # number of independent trials to run
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="parent directory for a new timestamped run folder")
    args = parser.parse_args()
    
    # Validate the arguments to ensure they are positive and meet the requirements for the experiment
    if args.n_eval < 1 or args.num_trials < 2:
        parser.error("n-eval must be positive; num-trials must be at least 2")
    if min(args.m1, args.m2) < 1:
        parser.error("m1 and m2 must be positive")
    if args.D < args.m1 + args.m2 + 2:
        parser.error("D must be at least (m1 + 1) + (m2 + 1)")
    if (len(args.sample_sizes) < 2 or min(args.sample_sizes) < 1
            or len(set(args.sample_sizes)) != len(args.sample_sizes)):
        parser.error("provide at least two distinct positive sample sizes")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    return args

# sample uniform points on the unit sphere S^m 
def sample_sphere(size, m, rng):
    """Uniform points on S^m in R^(m+1)."""
    points = rng.normal(size=(size, m + 1))
    return points / np.linalg.norm(points, axis=1, keepdims=True)

# sample independent points on the product of two unit spheres S^m1 x S^m2
def sample_product_spheres(size, m1, m2, rng):
    """Independent unit S^m1 and S^m2 factors, concatenated in R^(m1+m2+2)."""
    first = sample_sphere(size, m1, rng)
    second = sample_sphere(size, m2, rng)
    # Each factor has unit radius; do not normalize the concatenated vector.
    return np.concatenate((first, second), axis=1)

# compute W_2^2 using Euclidean squared cost and ot.emd2, which is the same as in 5.1.1
def w2_squared(source, target):
    """Same squared-cost construction and public POT call as sphere 5.1.1."""
    source_norms = np.einsum("ij,ij->i", source, source)
    target_norms = np.einsum("ij,ij->i", target, target)
    inner = np.einsum("ik,jk->ij", source, target, optimize=False)
    cost = source_norms[:, None] + target_norms[None, :] - 2.0 * inner
    np.maximum(cost, 0.0, out=cost)
    value, info = ot.emd2(
        np.full(len(source), 1.0 / len(source)),
        np.full(len(target), 1.0 / len(target)),
        cost,
        numItermax=1_000_000, log=True,
    )
    if info["warning"] or not np.isfinite(value):
        raise RuntimeError(f"OT solve failed: {info['warning']}")
    return float(max(value, 0.0))

# Compute unsmoothed W_2^2(mu_n,mu) on S^m using geodesic squared distance on the product spheres
def geodesic_w2_squared(source, target, m1, m2):
    """Product metric: theta_1^2 + theta_2^2 on the two unit-sphere factors."""
    split = m1 + 1
    end = split + m2 + 1
    inner_first = np.einsum("ik,jk->ij", source[:, :split], target[:, :split], optimize=False)
    inner_second = np.einsum("ik,jk->ij", source[:, split:end], target[:, split:end], optimize=False)
    np.clip(inner_first, -1.0, 1.0, out=inner_first)
    np.clip(inner_second, -1.0, 1.0, out=inner_second)
    cost = np.arccos(inner_first) ** 2 + np.arccos(inner_second) ** 2
    value, info = ot.emd2(
        np.full(len(source), 1.0 / len(source)),
        np.full(len(target), 1.0 / len(target)),
        cost,
        numItermax=1_000_000, log=True,
    )
    if info["warning"] or not np.isfinite(value):
        raise RuntimeError(f"OT solve failed: {info['warning']}")
    return float(max(value, 0.0))

def run_experiment(args, directory):
    """Stream measurements to CSV and retain all draws for reproducibility."""
    coordinate_dimension = args.m1 + args.m2 + 2
    sigmas = 0.6 * (np.arange(81) / 80) ** 2 # quadratic grid from 0 to 0.6, inclusive

    # Save all draws for reproducibility
    fields = ["trial", "n", "grid_index", "sigma",
              "smoothed_euclidean_w2_squared", "unsmoothed_euclidean_w2_squared",
              "unsmoothed_geodesic_w2_squared", "cost_and_ot_seconds"]

    
    with (directory / "measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for trial in range(1, args.num_trials + 1):  
            for n in args.sample_sizes:
                # Fresh independent samples for each (trial, n), reused across sigma. 
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, trial, n, 0])) 
                target = sample_product_spheres(args.n_eval, args.m1, args.m2, rng)
                empirical = sample_product_spheres(n, args.m1, args.m2, rng)
                np.savez(directory / f"samples_trial_{trial}_n{n}.npz",
                         empirical=empirical, target=target)

                # As in 5.1.1, sample, index, and noise streams have suffixes 0, 2, 1.
                index_rng = np.random.default_rng(np.random.SeedSequence([args.seed, trial, n, 2]))
                indices = index_rng.integers(0, n, size=args.n_eval)
                np.save(directory / f"indices_trial_{trial}_n{n}.npy", indices)
                noise_rng = np.random.default_rng(np.random.SeedSequence([args.seed, trial, n, 1]))
                noise = noise_rng.normal(size=(args.n_eval, args.D))
                np.save(directory / f"noise_trial_{trial}_n{n}.npy", noise)

                # compute unsmoothed W_2^2(mu_n,mu) on S^m using both Euclidean and geodesic squared distances
                euclidean_baseline = w2_squared(empirical, target)
                geodesic_baseline = geodesic_w2_squared(empirical, target, args.m1, args.m2)

                # Embed the empirical measure in R^D and pad with zeros to preserve both direct unsmoothed distances.
                embedded_target = np.pad(target, ((0, 0), (0, args.D - coordinate_dimension)))
                centers = np.pad(empirical[indices], ((0, 0), (0, args.D - coordinate_dimension)))
                print(f"Trial {trial}/{args.num_trials}, n={n}: "
                      f"Euclidean baseline={euclidean_baseline:.10g}, "
                      f"geodesic baseline={geodesic_baseline:.10g}", flush=True)

                for j, sigma in enumerate(sigmas):

                    # The same centers and Gaussian vectors are scaled across sigma.
                    source = centers + sigma * noise
                    started = perf_counter() # record the time taken to compute the cost and OT distance
                    
                    # compute W_2^2(k_sigma * mu_n , mu) using ot.emd2()
                    value = w2_squared(source, embedded_target) 
                    elapsed = perf_counter() - started
                    writer.writerow(dict(zip(fields, [trial, n, j, float(sigma), value,
                                                     euclidean_baseline, geodesic_baseline, elapsed])))
                    handle.flush()
                    print(f"  n={n}, sigma {j + 1}/81: {sigma:.6g}, "
                          f"W2^2={value:.10g} ({elapsed:.1f} sec)", flush=True)


def main():
    args = parse_args() # parse command-line arguments for experiment
    started = datetime.now().astimezone() # record the start time of the experiment
    sample_size_label = f"{min(args.sample_sizes)}-{max(args.sample_sizes)}x{len(args.sample_sizes)}"
    # Create a unique directory for the experiment based on parameters and timestamp
    directory = args.output_dir.resolve() / (
        f"S{args.m1}xS{args.m2}_D{args.D}_n{sample_size_label}_N{args.n_eval}_T{args.num_trials}_G81_"
        f"{started.strftime('%Y%m%d-%H%M%S-%f')}")
    directory.mkdir(parents=True, exist_ok=False)
  
   # Save the experiment configuration to a JSON file for reproducibility, including parameters, 
    # random seeds, and versions of libraries used  
    config = {
        **vars(args), "output_dir": str(args.output_dir.resolve()),
        "manifold": f"S^{args.m1} x S^{args.m2}",
        "intrinsic_dimension": args.m1 + args.m2, "ambient_dimension": args.D,
        "factor_dimensions": [args.m1, args.m2], "factor_radii": [1.0, 1.0],
        "coordinate_dimension": args.m1 + args.m2 + 2,
        "sigma_values": (0.6 * (np.arange(81) / 80) ** 2).tolist(),
        "num_sigmas": 81, "sigma_min": 0.0, "sigma_max": 0.6,
        "sigma_grid": "quadratic", "solver": "ot.emd2", "numItermax": 1_000_000,
        "smoothed_cost": "squared Euclidean",
        "unsmoothed_reference_costs": ["squared Euclidean", "squared product geodesic: theta_1^2 + theta_2^2"],
        "weights": "uniform on each discrete marginal",
        "randomness": "Fresh target, empirical, indices, and Gaussian noise per (trial, n); all reused across sigma.",
        "seed_policy": "Samples: SeedSequence([seed, trial, n, 0]); noise: SeedSequence([seed, trial, n, 1]); indices: SeedSequence([seed, trial, n, 2]); trials start at 1; target drawn before empirical.",
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
    try:
        run_experiment(args, directory)
    except BaseException:
        config["status"] = "incomplete"
        raise
    else:
        config["status"] = "complete"
    finally:
        config["run_ended_at"] = datetime.now().astimezone().isoformat()
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Finished. Run {directory / 'plot_results.cmd'} to create the plots.")


if __name__ == "__main__":
    main()
