"""Plot only the saved data beside this file; no OT computations are performed."""

import csv
import io
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

HERE = Path(__file__).resolve().parent
SIGMA_CLIP_LIMITS = (0.20, 0.30, 0.35, 0.40, 0.60)


def load_results(directory):
    config = json.loads((directory / "run_config.json").read_text(encoding="utf-8"))
    if config.get("status") != "complete":
        raise ValueError("This run is not marked complete; wait for the experiment to finish.")
    with (directory / "measurements.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    dimensions = config["dimensions"]
    has_explicit_suites = "suite_2_dimensions" in config
    suite_1_dimensions = config.get("suite_1_dimensions", dimensions)
    suite_2_dimensions = config.get("suite_2_dimensions", dimensions)
    sigmas = np.array(config["sigma_values"])
    trials = config["num_trials"]
    curves = np.full((len(dimensions), trials, len(sigmas)), np.nan)
    euclidean_baselines = np.full(trials, np.nan)
    geodesic_baselines = np.full(trials, np.nan)
    seen = set()
    for row in rows:

        d = dimensions.index(int(row["ambient_dimension"]))
        t, j = int(row["trial"]) - 1, int(row["grid_index"])
        key = (d, t, j)
        
        if key in seen or not (0 <= t < trials and 0 <= j < len(sigmas)):
            raise ValueError("Duplicate or invalid trial/grid key")
        seen.add(key)
        if not np.isclose(float(row["sigma"]), sigmas[j], rtol=1e-12, atol=0):
            raise ValueError("CSV sigma disagrees with the saved grid")
        
        curves[d, t, j] = float(row["smoothed_euclidean_w2_squared"])
        euclidean = float(row["unsmoothed_euclidean_w2_squared"])
        geodesic = float(row["unsmoothed_geodesic_w2_squared"])
        
        if ((np.isfinite(euclidean_baselines[t]) and euclidean != euclidean_baselines[t])
                or (np.isfinite(geodesic_baselines[t]) and geodesic != geodesic_baselines[t])):
            raise ValueError("An unsmoothed baseline changes within a trial")
        
        euclidean_baselines[t] = euclidean
        geodesic_baselines[t] = geodesic
    
    if (not np.isfinite(curves).all() or np.any(curves < 0)
            or not np.isfinite(euclidean_baselines).all()
            or not np.isfinite(geodesic_baselines).all()):
        raise ValueError("Missing or invalid distances")
    return (dimensions, suite_1_dimensions, suite_2_dimensions, has_explicit_suites,
            sigmas, curves, euclidean_baselines, geodesic_baselines)


def save_plot(fig, ax, directory, name, *, endpoint=False, ylabel=r"$W_2^2$",
              legend_loc="upper left", legend_fontsize=16):
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(labelsize=13)
    ax.set_axisbelow(True)
    ax.grid(True, linestyle="-" if endpoint else ":", alpha=0.25 if endpoint else 0.45)
    ax.legend(fontsize=14 if endpoint else legend_fontsize, loc=legend_loc)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        # Write a complete image before replacing an existing synced output.
        with tempfile.NamedTemporaryFile(dir=directory, suffix=f".{extension}", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            fig.savefig(temporary, format=extension, dpi=200)
            temporary.replace(directory / f"{name}.{extension}")
        finally:
            temporary.unlink(missing_ok=True)
    plt.close(fig)


def write_csv_if_changed(path, rows):
    """Keep an existing derived CSV untouched when its contents are unchanged."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=rows[0])
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    if not path.exists() or path.read_bytes() != content:
        path.write_bytes(content)


def save_average_curve_optima(directory, dimensions, sigmas, means):
    """Record the grid minimum of each trial-averaged curve."""
    # find the minimum of the mean W_2^2 curve for each dimension
    best_indices = np.argmin(means, axis=1)
    rows = []
    for i, dimension in enumerate(dimensions):
        j = int(best_indices[i])
        rows.append({
            "ambient_dimension": int(dimension),
            "grid_index": j,
            "sigma_amb": float(sigmas[j]),
            "minimum_mean_w2_squared": float(means[i, j]),
            "optimum_at_grid_boundary": int(j in (0, len(sigmas) - 1)),
        })
    path = directory / "optimal_bandwidths_from_average_curves.csv"
    write_csv_if_changed(path, rows)
    return np.asarray([row["sigma_amb"] for row in rows])


def save_trial_optima(directory, dimensions, sigmas, curves):
    """Find the minimum on each saved trial curve, including zero optima."""
    best_indices = np.argmin(curves, axis=2)
    rows = []
    for i, dimension in enumerate(dimensions):
        for t, j in enumerate(best_indices[i], start=1):
            j = int(j)  # np.argmin chooses the smallest sigma when costs tie.
            rows.append({
                "ambient_dimension": int(dimension),
                "trial": t,
                "grid_index": j,
                "sigma_amb": float(sigmas[j]),
                "minimum_w2_squared": float(curves[i, t - 1, j]),
                "optimum_at_grid_boundary": int(j in (0, len(sigmas) - 1)),
            })
    write_csv_if_changed(directory / "optimal_bandwidths_by_trial.csv", rows)
    return sigmas[best_indices]


def plot_results(directory=HERE):

    # load the saved results from the specified directory
    (dimensions, suite_1_dimensions, suite_2_dimensions, has_explicit_suites,
     sigmas, curves, euclidean_baselines, geodesic_baselines) = load_results(directory)
    
    # Compute the mean and standard deviation of the smoothed W_2^2 curves across trials for each dimension and sigma.
    means, sd = curves.mean(axis=1), curves.std(axis=1, ddof=1)
    sigma_amb = save_average_curve_optima(directory, dimensions, sigmas, means)
    trial_sigma_amb = save_trial_optima(directory, dimensions, sigmas, curves)
    dimension_to_index = {dimension: i for i, dimension in enumerate(dimensions)}
    suite_1_indices = np.asarray([dimension_to_index[d] for d in suite_1_dimensions])
    suite_2_indices = np.asarray([dimension_to_index[d] for d in suite_2_dimensions])
    
    # for each specified sigma clipping limit, plot the smoothed W_2^2 curves for Suite 1 dimensions
    # along with the unsmoothed Euclidean and geodesic baselines, and save the plots to the specified directory
    for sigma_clipped in SIGMA_CLIP_LIMITS:
        keep = (sigmas >= 0.0) & (sigmas <= sigma_clipped)
        
        if np.count_nonzero(keep) < 2:
            raise ValueError(f"Fewer than two saved sigma points lie in [0, {sigma_clipped}]")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        for i in suite_1_indices:
            dimension = dimensions[i]
            line, = ax.plot(sigmas[keep], means[i, keep], linewidth=2,
                            label=rf"$D = {dimension}$")
            ax.fill_between(sigmas[keep], means[i, keep] - sd[i, keep],
                            means[i, keep] + sd[i, keep],
                            color=line.get_color(), alpha=0.15)
        for baselines, color, label in (
            (euclidean_baselines, "black", r"$W_2^2(\widehat{\mu}_n,\mu)$"),
            (geodesic_baselines, "dimgray", r"$W_{2,M}^2(\widehat{\mu}_n,\mu)$"),
        ):
            baseline_mean = baselines.mean()
            baseline_sd = baselines.std(ddof=1) if len(baselines) > 1 else 0.0
            ax.axhspan(baseline_mean - baseline_sd, baseline_mean + baseline_sd,
                       color=color, alpha=0.12, linewidth=0)
            ax.axhline(baseline_mean, color=color, linestyle="--", linewidth=2,
                       label=label)
        ax.set_xlabel(r"Bandwidth $\sigma$", fontsize=18)
        ax.set_xlim(0.0, sigma_clipped)
        sigma_formatter = ScalarFormatter(useOffset=False)
        sigma_formatter.set_scientific(False)
        ax.xaxis.set_major_formatter(sigma_formatter)
        limit_label = f"{sigma_clipped:.2f}".replace(".", "p")
        save_plot(fig, ax, directory,
                  f"w2_squared_vs_sigma_overlay_sigma_0_to_{limit_label}",
                  legend_fontsize=15)
   
    # sanity check
    fig, ax = plt.subplots(figsize=(7, 6))
    order = suite_1_indices[np.argsort(np.asarray(dimensions)[suite_1_indices])]
    x = np.asarray(dimensions)[order]
    for j, color in [(0, "tab:blue"), (-1, "tab:orange")]:
        ax.errorbar(x, means[order, j], yerr=sd[order, j], fmt="o-", color=color,
                    markersize=7, capsize=5, linewidth=2,
                    label=rf"$\sigma={sigmas[j]:.4g}$")
    ax.set_xticks(x)
    ax.set_xlabel(r"Ambient dimension $D$", fontsize=18)
    save_plot(fig, ax, directory, "w2_squared_endpoint_comparison", endpoint=True)

    # Fit a log-log regression to the average-curve optima in Suite 2 dimensions, and save the results to a CSV file.
    order = suite_2_indices[np.argsort(np.asarray(dimensions)[suite_2_indices])]
    sorted_dimensions = np.asarray(dimensions)[order]
    sorted_sigma_amb = sigma_amb[order]
    positive = sorted_sigma_amb > 0 # log-log regression is only valid for positive optima
    omitted = int(np.count_nonzero(~positive))

    # sanity check: if there are fewer than two positive optima, the regression cannot be performed
    regression = {
        "fit_model": "log(sigma_amb) = slope * log(D) + intercept",
        "suite": 2 if has_explicit_suites else "legacy run: all available dimensions",
        "num_positive_points": int(np.count_nonzero(positive)),
        "zero_optima_omitted": omitted,
        "slope": "",
        "intercept": "",
        "r_squared": "",
        "fit_status": "fewer than two positive optima",
    }
    if np.any(positive):
        # log-log regression 
        fig, ax = plt.subplots(figsize=(7, 6))
        fit_dimensions = sorted_dimensions[positive]
        fit_sigma_amb = sorted_sigma_amb[positive]
        ax.scatter(fit_dimensions, fit_sigma_amb, marker="D", color="navy", s=70,
                   label=r"Minimum of mean $W_2^2$ curve")
        if len(fit_dimensions) >= 2:
            log_dimensions = np.log(fit_dimensions)
            log_sigma_amb = np.log(fit_sigma_amb)
            slope, intercept = np.polyfit(log_dimensions, log_sigma_amb, 1)
            fitted_log_sigma = slope * log_dimensions + intercept
            residual = log_sigma_amb - fitted_log_sigma
            total = np.sum((log_sigma_amb - log_sigma_amb.mean()) ** 2)
            r_squared = 1.0 - np.sum(residual ** 2) / total if total > 0 else 1.0
            
            # plot the fitted line on the log-log scale
            fit_x = np.geomspace(fit_dimensions.min(), fit_dimensions.max(), 200)
            fit_y = np.exp(intercept) * fit_x ** slope
            ax.plot(fit_x, fit_y, color="red", linewidth=2,
                    label=rf"Fit: slope={slope:.4f}, $R^2$={r_squared:.4f}")
            regression.update(
                slope=float(slope), intercept=float(intercept),
                r_squared=float(r_squared), fit_status="ok")

        # log-log scale for x and y axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Ambient dimension $D$", fontsize=18)
        save_plot(fig, ax, directory, "sigma_amb_vs_D_loglog",
                  ylabel=r"Optimal bandwidth $\widehat{\sigma}_{amb}$",
                  legend_loc="upper right")
    write_csv_if_changed(directory / "sigma_amb_loglog_regression.csv", [regression])

    # Pool all positive trial-level optima in Suite 2. Every (trial, D)
    # contributes one point, so each dimension appears once per trial.
    trial_dimensions = np.repeat(np.asarray(dimensions)[suite_2_indices], curves.shape[1])
    trial_bandwidths = trial_sigma_amb[suite_2_indices].ravel()
    trial_positive = trial_bandwidths > 0
    fit_dimensions = trial_dimensions[trial_positive]
    fit_bandwidths = trial_bandwidths[trial_positive]
    trial_regression = {
        "fit_model": "log(sigma_amb(trial,D)) = slope * log(D) + intercept",
        "suite": 2 if has_explicit_suites else "legacy run: all available dimensions",
        "num_recorded_points": len(trial_bandwidths),
        "num_positive_points": len(fit_bandwidths),
        "zero_optima_omitted": int(np.count_nonzero(~trial_positive)),
        "slope": "", "intercept": "", "r_squared": "",
        "fit_status": "fewer than two positive optima at distinct dimensions",
    }
    if len(fit_bandwidths):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(fit_dimensions, fit_bandwidths, s=48, color="tab:blue",
                   alpha=0.65, label=r"Positive trial-level $\widehat{\sigma}_{amb}$")
        if len(np.unique(fit_dimensions)) >= 2:
            log_dimensions = np.log(fit_dimensions)
            log_bandwidths = np.log(fit_bandwidths)
            slope, intercept = np.polyfit(log_dimensions, log_bandwidths, 1)
            fitted_log_bandwidths = slope * log_dimensions + intercept
            residual = log_bandwidths - fitted_log_bandwidths
            total = np.sum((log_bandwidths - log_bandwidths.mean()) ** 2)
            r_squared = 1.0 - np.sum(residual ** 2) / total if total > 0 else 1.0
            fit_x = np.geomspace(fit_dimensions.min(), fit_dimensions.max(), 200)
            ax.plot(fit_x, np.exp(intercept) * fit_x ** slope,
                    color="red", linewidth=2,
                    label=rf"Pooled fit: slope={slope:.4f}, $R^2$={r_squared:.4f}")
            trial_regression.update(slope=float(slope), intercept=float(intercept),
                                    r_squared=float(r_squared), fit_status="ok")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Ambient dimension $D$", fontsize=18)
        save_plot(fig, ax, directory, "sigma_amb_by_trial_vs_D_loglog",
                  ylabel=r"Trial optimal bandwidth $\widehat{\sigma}_{amb}$",
                  legend_loc="upper right")
    write_csv_if_changed(
        directory / "sigma_amb_trial_pooled_loglog_regression.csv", [trial_regression])
    fit_scope = "Suite 2" if has_explicit_suites else "legacy-run dimensions"
    print(
        f"Saved five clipped overlays, endpoint comparison, average-curve optima, "
        f"trial optima, and both {fit_scope} log-log regressions in {directory}. "
        f"Pooled trial slope: {trial_regression['slope']}; "
        f"R^2: {trial_regression['r_squared']}. "
        f"Zero trial optima omitted: {trial_regression['zero_optima_omitted']}/"
        f"{trial_regression['num_recorded_points']}"
    )


if __name__ == "__main__":
    plot_results()
