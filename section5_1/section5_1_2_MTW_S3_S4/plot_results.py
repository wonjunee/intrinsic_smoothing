"""Plot saved Experiment 5.1.2 results without importing or running an OT solver."""

import argparse
import csv
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def load_results(directory):
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    if config["status"] != "complete":
        raise ValueError("The run is incomplete; refusing to plot partial averages.")
    sizes = np.array(config["sample_sizes"])
    sigmas = np.array(config["sigma_values"])
    if not np.array_equal(sigmas, 0.6 * (np.arange(81) / 80) ** 2):
        raise ValueError("Expected the specified 81-point quadratic bandwidth grid.")
    curves = np.full((len(sizes), config["num_trials"], len(sigmas)), np.nan)
    baselines = np.full((len(sizes), config["num_trials"], 2), np.nan)
    size_index = {n: i for i, n in enumerate(sizes)}
    with (directory / "measurements.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            i, t, j = size_index[int(row["n"])], int(row["trial"]) - 1, int(row["grid_index"])
            if not (0 <= t < curves.shape[1] and 0 <= j < len(sigmas)):
                raise ValueError("Invalid trial or grid index.")
            if np.isfinite(curves[i, t, j]) or float(row["sigma"]) != sigmas[j]:
                raise ValueError("Duplicate observation or inconsistent bandwidth.")
            values = [float(row[f"unsmoothed_{metric}_w2_squared"])
                      for metric in ("euclidean", "geodesic")]
            if np.isfinite(baselines[i, t]).all() and not np.array_equal(baselines[i, t], values):
                raise ValueError("Unsmoothed baselines change across bandwidths.")
            baselines[i, t] = values
            curves[i, t, j] = float(row["smoothed_euclidean_w2_squared"])
    if any(not np.isfinite(a).all() or np.any(a < 0) for a in (curves, baselines)):
        raise ValueError("Missing or invalid OT values.")
    return sizes, sigmas, curves, baselines


def save_plot(fig, ax, directory, name, xlabel, ylabel=r"$W_2^2$", *, loglog=False, close=True):
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(True, which="both" if loglog else "major", linestyle=":",
            alpha=0.25 if loglog else 0.45)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        # Finish each image before replacing a synced output, as in 5.1.1.
        with tempfile.NamedTemporaryFile(dir=directory, suffix=f".{extension}", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, format=extension, dpi=200)
            temporary.replace(directory / f"{name}.{extension}")
        finally:
            temporary.unlink(missing_ok=True)
    if close:
        plt.close(fig)


def plot_curves(directory, sizes, sigmas, curves, baselines, *, overlay_only=False):
    means, sd = curves.mean(axis=1), curves.std(axis=1, ddof=1)
    baseline_means, baseline_sd = baselines.mean(axis=1), baselines.std(axis=1, ddof=1)
    groups = [(f"w2_squared_vs_sigma_n_{n}", [i]) for i, n in enumerate(sizes)]
    if overlay_only:
        groups = []
    groups.append(("w2_squared_vs_sigma_overlay", list(range(len(sizes)))))
    for name, indices in groups:
        overlay = name == "w2_squared_vs_sigma_overlay"
        for limit in (0.20, 0.30, 0.40, 0.60):
            # Overlays include zero on a linear axis; individual-n plots stay logarithmic.
            # Interpolate only the display endpoint; saved OT distances are unchanged.
            keep = (sigmas < limit) & ((sigmas >= 0) if overlay else (sigmas > 0))
            x = np.append(sigmas[keep], limit)
            fig, ax = plt.subplots(figsize=(9, 6.5))
            for i in indices:
                color = f"C{i % 10}"
                mean = np.interp(x, sigmas, means[i])
                spread = np.interp(x, sigmas, sd[i])
                ax.plot(x, mean, color=color, linewidth=2,
                        label=rf"Smoothed, $n={sizes[i]}$")
                ax.fill_between(x, mean - spread, mean + spread, color=color, alpha=0.15)
                for k, (metric, style) in enumerate((("Euclidean", "--"), ("Geodesic", ":"))):
                    baseline, baseline_spread = baseline_means[i, k], baseline_sd[i, k]
                    ax.axhspan(baseline - baseline_spread, baseline + baseline_spread,
                               color=color, alpha=0.08, linewidth=0)
                    ax.axhline(baseline, color=color, linestyle=style, linewidth=1.7,
                               label=rf"{metric}, $n={sizes[i]}$")
            # Keep each n's three colored entries together, beside the plotting area.
            ax.legend(ncol=1, fontsize=10, loc="center left",
                      bbox_to_anchor=(1.02, 0.5), borderaxespad=0,
                      frameon=False, handlelength=2.2, labelspacing=0.45)
            ax.set_xscale("linear" if overlay else "log")
            ax.set_xlim(0 if overlay else x[0], limit)
            if overlay:
                ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            limit_label = f"{limit:.2f}".replace(".", "p")
            clipped_name = f"{name}_sigma_0_to_{limit_label}"
            save_plot(fig, ax, directory, clipped_name, r"Bandwidth $\sigma$",
                      loglog=not overlay, close=False)
            ax.get_legend().remove()
            save_plot(fig, ax, directory, clipped_name + "_no_legend",
                      r"Bandwidth $\sigma$", loglog=not overlay)
            if limit == 0.60:
                # Keep the original full-range filenames current for existing links.
                for suffix in ("", "_no_legend"):
                    for extension in ("png", "pdf"):
                        (directory / f"{name}{suffix}.{extension}").write_bytes(
                            (directory / f"{clipped_name}{suffix}.{extension}").read_bytes())



def plot_regression(directory, name, sizes, bandwidths, label, m, *, trial_points=False):
    """Unweighted OLS on positive optima; zeros remain in the saved optimum tables."""
    positive = bandwidths > 0
    x, y = sizes[positive], bandwidths[positive]
    summary = {
        "method": name, "fit_model": "log(sigma_amb) = slope * log(n) + intercept",
        "num_recorded_points": len(bandwidths), "num_positive_points": len(y),
        "zero_optima_omitted": int((~positive).sum()),
        "slope": None, "intercept": None, "r_squared": None,
        "reference_slope": -1.0 / m, "reference_intercept": None,
        "fit_status": "fewer than two distinct sample sizes with positive optima",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    if trial_points:
        ax.scatter(x, y, s=30, color="tab:blue", alpha=0.65,
                   label=r"Trial-level $\hat{\sigma}_{\mathrm{amb}}$")
    elif len(y):
        ax.scatter(x, y, marker="D", s=65, color="navy", zorder=3, label=label)
    if len(np.unique(x)) >= 2:
        log_x, log_y = np.log(x), np.log(y)
        slope, intercept = np.polyfit(log_x, log_y, 1)
        residual = log_y - (slope * log_x + intercept)
        total = np.sum((log_y - log_y.mean()) ** 2)
        r_squared = float(1 - np.sum(residual ** 2) / total) if total > 0 else None
        fit_x = np.geomspace(x.min(), x.max(), 200)
        ax.plot(fit_x, np.exp(intercept) * fit_x ** slope, color="red", linewidth=2,
                label=f"Fitted slope = {slope:.4f}")
        # Set the reference height by least squares in log space at fixed slope -1/m.
        reference_slope = -1.0 / m
        reference_intercept = float(np.mean(log_y - reference_slope * log_x))
        ax.plot(fit_x, np.exp(reference_intercept) * fit_x ** reference_slope,
                color="black", linestyle="--", linewidth=2,
                label=rf"Reference slope $-1/m={reference_slope:.4f}$")
        summary["reference_intercept"] = reference_intercept
        summary.update(slope=float(slope), intercept=float(intercept),
                       r_squared=r_squared, fit_status="ok")
    else:
        ax.text(0.5, 0.5, "Insufficient positive optima for a fit",
                ha="center", transform=ax.transAxes)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if len(y):
        ax.legend(fontsize=11, loc="upper right")
    else:
        ax.set_ylim(1e-5, 1)
    save_plot(fig, ax, directory, name, r"Sample size $n$",
              r"Optimal bandwidth $\hat{\sigma}_{\mathrm{amb}}$", loglog=True)
    print(f"{name}: slope={summary['slope']}; zero optima omitted="
          f"{summary['zero_optima_omitted']}/{len(bandwidths)}")
    return summary


def plot_results(directory=HERE):
    directory = Path(directory).resolve()
    sizes, sigmas, curves, baselines = load_results(directory)
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    m = int(config["intrinsic_dimension"])
    plot_curves(directory, sizes, sigmas, curves, baselines)
    means = curves.mean(axis=1)
    mean_indices, trial_indices = means.argmin(axis=1), curves.argmin(axis=2)
    # np.argmin resolves ties toward the smallest bandwidth on this increasing grid.
    mean_rows, trial_rows = [], []
    for i, n in enumerate(sizes):
        j = int(mean_indices[i])
        mean_rows.append(dict(n=int(n), grid_index=j, sigma_amb=float(sigmas[j]),
                              minimum_mean_w2_squared=float(means[i, j]),
                              optimum_at_grid_boundary=int(j in (0, len(sigmas) - 1))))
        for t, j in enumerate(trial_indices[i]):
            j = int(j)
            trial_rows.append(dict(n=int(n), trial=t + 1, grid_index=j,
                                   sigma_amb=float(sigmas[j]), minimum_w2_squared=float(curves[i, t, j]),
                                   optimum_at_grid_boundary=int(j in (0, len(sigmas) - 1))))
    write_csv(directory / "optimal_bandwidths_from_average_curves.csv", mean_rows)
    write_csv(directory / "optimal_bandwidths_by_trial.csv", trial_rows)
    summaries = [
        plot_regression(directory, "sigma_amb_vs_n_loglog", sizes,
                        sigmas[mean_indices], "Mean-curve minima", m),
        plot_regression(directory, "sigma_amb_by_trial_vs_n_loglog",
                        np.repeat(sizes, curves.shape[1]), sigmas[trial_indices].ravel(),
                        "Trial minima", m, trial_points=True),
    ]
    write_csv(directory / "loglog_regressions.csv", summaries)
    print(f"Plots and optimum tables saved to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=HERE, help="completed run directory containing measurements.csv")
    plot_results(parser.parse_args().directory)
