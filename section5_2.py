#!/usr/bin/env python3
"""Recreate Section 5.2 figures from saved CSV output.

This script performs no sampling, decoder evaluation, or optimal transport
computation.  It reads the CSV and metadata files written by
``latent_smoothing_section_5_2.py`` and recreates the paper figures.  By
default, W_2^2-versus-sigma figures use ``figsize=(7.5, 5.0)``.

Typical usage
-------------

Recreate every figure in the original results directory::

    python plot_latent_smoothing_results.py \
        --results-dir explicit_global_encoder_decoder_results

Write the figures to a separate directory::

    python plot_latent_smoothing_results.py \
        --results-dir explicit_global_encoder_decoder_results \
        --output-dir explicit_global_encoder_decoder_replots

Only recreate W_2^2-versus-sigma figures::

    python plot_latent_smoothing_results.py \
        --results-dir explicit_global_encoder_decoder_results \
        --only w2

Created with ChatGPT
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


Array = np.ndarray
PARAMETER_KEYS = ("rho", "delta", "lorth", "latent_dim")
MANIFOLD_LATEX = {
    "sphere": r"$\mathbb S^2$",
    "product": r"$\mathbb S^1\times\mathbb S^2$",
}
INTRINSIC_DIMENSION = {"sphere": 2, "product": 3}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required data file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_metadata(results_dir: Path) -> dict[str, object]:
    path = results_dir / "run_metadata.json"
    if not path.exists():
        print(
            f"WARNING: {path.name} is missing; using plotting defaults.",
            flush=True,
        )
        return {"arguments": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metadata_argument(
    metadata: dict[str, object],
    name: str,
    default: float,
) -> float:
    arguments = metadata.get("arguments", {})
    if not isinstance(arguments, dict):
        return default
    value = arguments.get(name, default)
    return float(value)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 19,
            "legend.fontsize": 12,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "lines.linewidth": 2.4,
            "lines.markersize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(
    figure: plt.Figure,
    base_path: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for suffix in formats:
        output = base_path.with_suffix(f".{suffix}")
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.unlink(missing_ok=True)
        figure.savefig(
            temporary,
            format=suffix,
            bbox_inches="tight",
        )
        if suffix == "pdf":
            with temporary.open("rb") as handle:
                handle.seek(max(0, temporary.stat().st_size - 1024))
                if b"%%EOF" not in handle.read():
                    temporary.unlink(missing_ok=True)
                    raise OSError(f"Incomplete PDF figure: {output}")
        temporary.replace(output)
        outputs.append(output)
    plt.close(figure)
    return outputs


def t_confidence_halfwidth(
    samples: Array,
    confidence_level: float,
    axis: int = 0,
) -> Array:
    samples = np.asarray(samples, dtype=float)
    sample_count = samples.shape[axis]
    output_shape = samples.shape[:axis] + samples.shape[axis + 1 :]
    if sample_count <= 1:
        return np.zeros(output_shape, dtype=float)
    standard_error = np.std(samples, axis=axis, ddof=1) / np.sqrt(sample_count)
    quantile = float(
        student_t.ppf(0.5 + 0.5 * confidence_level, sample_count - 1)
    )
    return quantile * standard_error


def parameter_label(key: str, value: float) -> str:
    if key == "rho":
        return rf"$\rho_\parallel={value:g}$"
    if key == "delta":
        return rf"$\delta={value:g}$"
    if key == "lorth":
        return rf"$L_{{\rm orth}}={value:g}$"
    if key == "latent_dim":
        return rf"$d={int(value)}$"
    raise KeyError(key)


def load_summary_curves(
    results_dir: Path,
) -> dict[tuple[str, str], dict[str, Array]]:
    rows = read_csv(results_dir / "explicit_w2_curves.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["manifold"], row["parameter"])].append(row)

    result: dict[tuple[str, str], dict[str, Array]] = {}
    for key, selected in grouped.items():
        values = np.asarray(
            sorted({float(row["parameter_value"]) for row in selected})
        )
        sigma = np.asarray(sorted({float(row["sigma"]) for row in selected}))
        value_index = {value: index for index, value in enumerate(values)}
        sigma_index = {value: index for index, value in enumerate(sigma)}
        shape = (len(values), len(sigma))
        mean = np.full(shape, np.nan)
        std = np.full(shape, np.nan)
        ci = np.full(shape, np.nan)
        for row in selected:
            i = value_index[float(row["parameter_value"])]
            j = sigma_index[float(row["sigma"])]
            mean[i, j] = float(row["w2_squared_mean"])
            std[i, j] = float(row["w2_squared_std"])
            ci[i, j] = float(row["w2_squared_ci_halfwidth"])
        if np.any(~np.isfinite(mean)):
            raise ValueError(f"Incomplete curve table for {key}.")
        result[key] = {
            "values": values,
            "sigma": sigma,
            "mean": mean,
            "std": std,
            "ci": ci,
        }
    return result


def load_run_curves(
    results_dir: Path,
) -> dict[tuple[str, str], dict[str, Array]]:
    rows = read_csv(results_dir / "explicit_w2_curves_by_run.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["manifold"], row["parameter"])].append(row)

    result: dict[tuple[str, str], dict[str, Array]] = {}
    for key, selected in grouped.items():
        repetitions = np.asarray(
            sorted({int(row["repetition"]) for row in selected})
        )
        values = np.asarray(
            sorted({float(row["parameter_value"]) for row in selected})
        )
        sigma = np.asarray(sorted({float(row["sigma"]) for row in selected}))
        repetition_index = {
            value: index for index, value in enumerate(repetitions)
        }
        value_index = {value: index for index, value in enumerate(values)}
        sigma_index = {value: index for index, value in enumerate(sigma)}
        curves = np.full(
            (len(repetitions), len(values), len(sigma)),
            np.nan,
        )
        for row in selected:
            i = repetition_index[int(row["repetition"])]
            j = value_index[float(row["parameter_value"])]
            k = sigma_index[float(row["sigma"])]
            curves[i, j, k] = float(row["w2_squared"])
        if np.any(~np.isfinite(curves)):
            raise ValueError(f"Incomplete run-level curve table for {key}.")
        result[key] = {
            "repetitions": repetitions,
            "values": values,
            "sigma": sigma,
            "curves": curves,
        }
    return result


def load_optima(
    results_dir: Path,
) -> dict[tuple[str, str], dict[str, Array]]:
    rows = read_csv(results_dir / "explicit_optimal_sigma.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["manifold"], row["parameter"])].append(row)

    result: dict[tuple[str, str], dict[str, Array]] = {}
    for key, selected in grouped.items():
        selected.sort(key=lambda row: float(row["parameter_value"]))

        def optional_column(name: str) -> Array:
            values: list[float] = []
            for row in selected:
                raw = row.get(name, "")
                values.append(float(raw) if raw not in (None, "") else np.nan)
            return np.asarray(values, dtype=float)

        result[key] = {
            "values": optional_column("parameter_value"),
            "mean": optional_column("optimal_sigma_mean"),
            "std": optional_column("optimal_sigma_std"),
            "ci": optional_column("optimal_sigma_ci_halfwidth"),
            "descriptive": optional_column("descriptive_prediction"),
            "dominant": optional_column("dominant_term_prediction"),
        }
    return result


def load_normalized_rho(
    results_dir: Path,
) -> dict[str, dict[str, Array]]:
    rows = read_csv(results_dir / "explicit_normalized_optima.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["parameter"] == "rho":
            grouped[row["manifold"]].append(row)
    result: dict[str, dict[str, Array]] = {}
    for manifold, selected in grouped.items():
        selected.sort(key=lambda row: float(row["parameter_value"]))
        result[manifold] = {
            "values": np.asarray(
                [float(row["parameter_value"]) for row in selected]
            ),
            "mean": np.asarray(
                [float(row["normalized_optimal_sigma_mean"]) for row in selected]
            ),
            "std": np.asarray(
                [float(row["normalized_optimal_sigma_std"]) for row in selected]
            ),
            "ci": np.asarray(
                [
                    float(row["normalized_optimal_sigma_ci_halfwidth"])
                    for row in selected
                ]
            ),
        }
    return result


def load_fit_parameters(
    results_dir: Path,
) -> dict[tuple[str, str], float]:
    rows = read_csv(results_dir / "explicit_fit_diagnostics.csv")
    return {
        (row["manifold"], row["fit"]): float(row["nuisance_value"])
        for row in rows
    }


def centered_curve_summary(
    curves: Array,
    confidence_level: float,
) -> tuple[Array, Array]:
    centered = curves - curves[..., [0]]
    return (
        np.mean(centered, axis=0),
        t_confidence_halfwidth(centered, confidence_level, axis=0),
    )


def refined_grid_minimum(
    x_values: Array,
    y_values: Array,
) -> tuple[float, float]:
    """Match the local quadratic minimizer used by the simulation script."""
    index = int(np.argmin(y_values))
    if index == 0 or index == len(x_values) - 1:
        return float(x_values[index]), float(y_values[index])
    local_x = x_values[index - 1 : index + 2]
    local_y = y_values[index - 1 : index + 2]
    curvature, linear, constant = np.polyfit(local_x, local_y, deg=2)
    if curvature <= 0.0:
        return float(x_values[index]), float(y_values[index])
    vertex = float(-linear / (2.0 * curvature))
    if not local_x[0] <= vertex <= local_x[-1]:
        return float(x_values[index]), float(y_values[index])
    return vertex, float(curvature * vertex**2 + linear * vertex + constant)


def mean_runwise_optimum(sigma: Array, curves: Array) -> float:
    """Average the refined minimizer computed separately in each run."""
    flattened = np.asarray(curves, dtype=float).reshape(-1, len(sigma))
    optima = [refined_grid_minimum(sigma, curve)[0] for curve in flattened]
    return float(np.mean(optima))


def local_sigma_limit(
    sigma: Array,
    mean_optima: Array,
    display_sigma_max: float,
) -> float:
    optima = np.asarray(mean_optima, dtype=float)
    positive = optima[(optima > 0.0) & np.isfinite(optima)]
    local = positive[positive <= 0.25 * display_sigma_max]
    if local.size == 0:
        local = positive
    minimum_index = min(8, len(sigma) - 1)
    minimum_window = float(sigma[minimum_index])
    if local.size == 0:
        return min(display_sigma_max, max(minimum_window, 0.05))
    return min(
        display_sigma_max,
        max(minimum_window, 1.5 * float(np.max(local))),
    )


def plot_w2_curves(
    manifold: str,
    key: str,
    summary_curves: dict[tuple[str, str], dict[str, Array]],
    run_curves: dict[tuple[str, str], dict[str, Array]],
    optima: dict[tuple[str, str], dict[str, Array]],
    output_dir: Path,
    display_sigma_max: float,
    confidence_level: float,
    figure_size: tuple[float, float],
    include_ambient: bool,
    formats: tuple[str, ...],
) -> list[Path]:
    summary = summary_curves[(manifold, key)]
    raw = run_curves[(manifold, key)]
    optimum = optima[(manifold, key)]
    ambient_summary = summary_curves[(manifold, "ambient_smoothing")]
    ambient_raw = run_curves[(manifold, "ambient_smoothing")]

    values = summary["values"]
    sigma = summary["sigma"]
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(values)))
    figure, axis = plt.subplots(figsize=figure_size)

    for index, (value, color) in enumerate(zip(values, colors)):
        mean = summary["mean"][index]
        ci = summary["ci"][index]
        axis.plot(sigma, mean, color=color, label=parameter_label(key, value))
        axis.fill_between(
            sigma,
            mean - ci,
            mean + ci,
            color=color,
            alpha=0.08,
            linewidth=0.0,
        )
        sigma_star = float(optimum["mean"][index])
        axis.plot(
            sigma_star,
            np.interp(sigma_star, sigma, mean),
            marker="o",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )

    ambient_mean = ambient_summary["mean"][0]
    ambient_ci = ambient_summary["ci"][0]
    ambient_sigma_star = mean_runwise_optimum(
        sigma,
        ambient_raw["curves"],
    )
    if include_ambient:
        axis.fill_between(
            sigma,
            ambient_mean - ambient_ci,
            ambient_mean + ambient_ci,
            color="black",
            alpha=0.08,
            linewidth=0.0,
            zorder=2,
        )
        axis.plot(
            sigma,
            ambient_mean,
            color="black",
            linestyle="--",
            linewidth=3.0,
            label="Ambient",
            zorder=3,
        )
        axis.plot(
            ambient_sigma_star,
            np.interp(ambient_sigma_star, sigma, ambient_mean),
            marker="o",
            markersize=5.5,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )

    centered_mean, centered_ci = centered_curve_summary(
        raw["curves"], confidence_level
    )
    centered_ambient_mean, centered_ambient_ci = centered_curve_summary(
        ambient_raw["curves"], confidence_level
    )
    centered_ambient_mean = centered_ambient_mean[0]
    centered_ambient_ci = centered_ambient_ci[0]

    inset = axis.inset_axes([0.15, 0.55, 0.46, 0.38])
    inset.set_zorder(20)
    zoom_sigma = local_sigma_limit(
        sigma,
        optimum["mean"],
        display_sigma_max,
    )
    zoom_mask = sigma <= zoom_sigma
    for index, color in enumerate(colors):
        inset.plot(
            sigma[zoom_mask],
            centered_mean[index, zoom_mask],
            color=color,
            linewidth=1.6,
        )
        inset.fill_between(
            sigma[zoom_mask],
            centered_mean[index, zoom_mask] - centered_ci[index, zoom_mask],
            centered_mean[index, zoom_mask] + centered_ci[index, zoom_mask],
            color=color,
            alpha=0.06,
            linewidth=0.0,
        )
        sigma_star = float(optimum["mean"][index])
        if sigma_star <= zoom_sigma:
            inset.plot(
                sigma_star,
                np.interp(sigma_star, sigma, centered_mean[index]),
                marker="o",
                markersize=4.0,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
            )

    if include_ambient:
        inset.plot(
            sigma[zoom_mask],
            centered_ambient_mean[zoom_mask],
            color="black",
            linestyle="--",
            linewidth=1.6,
            zorder=5,
        )
        inset.fill_between(
            sigma[zoom_mask],
            centered_ambient_mean[zoom_mask]
            - centered_ambient_ci[zoom_mask],
            centered_ambient_mean[zoom_mask]
            + centered_ambient_ci[zoom_mask],
            color="black",
            alpha=0.05,
            linewidth=0.0,
        )
        if ambient_sigma_star <= zoom_sigma:
            inset.plot(
                ambient_sigma_star,
                np.interp(
                    ambient_sigma_star,
                    sigma,
                    centered_ambient_mean,
                ),
                marker="o",
                markersize=4.0,
                markerfacecolor="black",
                markeredgecolor="white",
                markeredgewidth=0.6,
                linestyle="none",
                zorder=6,
            )

    inset.axhline(0.0, color="0.35", linestyle=":", linewidth=1.2)
    inset.set_xlim(0.0, zoom_sigma)
    negative_depths = [
        -float(np.min(centered_mean[index, zoom_mask]))
        for index in range(len(values))
        if float(np.min(centered_mean[index, zoom_mask])) < 0.0
    ]
    if negative_depths:
        linear_threshold = max(
            1.0e-10,
            0.5 * float(np.median(negative_depths)),
        )
        inset.set_yscale(
            "symlog",
            linthresh=linear_threshold,
            linscale=1.0,
            base=10,
        )
        inset.text(
            0.03,
            0.93,
            "symlog scale",
            transform=inset.transAxes,
            fontsize=8,
            ha="left",
            va="top",
        )
    inset.set_xlabel(r"$\sigma$", fontsize=10, labelpad=0)
    inset.set_ylabel(r"$\Delta W_2^2$", fontsize=10, labelpad=1)
    inset.tick_params(axis="both", labelsize=8)
    inset.grid(alpha=0.18)
    inset.set_facecolor("white")
    inset.patch.set_alpha(1.0)

    display_mask = sigma <= display_sigma_max
    y_min = float(np.min((summary["mean"] - summary["ci"])[:, display_mask]))
    y_max = float(np.max((summary["mean"] + summary["ci"])[:, display_mask]))
    if include_ambient:
        y_min = min(
            y_min,
            float(np.min((ambient_mean - ambient_ci)[display_mask])),
        )
        y_max = max(
            y_max,
            float(np.max((ambient_mean + ambient_ci)[display_mask])),
        )
    margin = 0.05 * max(y_max - y_min, np.finfo(float).eps)
    axis.set_xlim(0.0, display_sigma_max)
    axis.set_ylim(min(0.0, y_min - margin), y_max + margin)
    axis.set_xlabel(r"Bandwidth $\sigma$")
    axis.set_ylabel(r"$W_2^2$")
    axis.grid(alpha=0.25)
    axis.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
    )
    figure.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    suffix = "with_ambient" if include_ambient else "latent_only"
    return save_figure(
        figure,
        output_dir
        / manifold
        / f"{manifold}_explicit_w2_vs_sigma_{key}_{suffix}",
        formats,
    )


def rho_slope_and_ci(
    results_dir: Path,
    manifold: str,
    values: Array,
    normalized_mean: Array,
    confidence_level: float,
) -> tuple[float, float]:
    path = results_dir / "explicit_rho_slope_by_run.csv"
    if path.exists():
        rows = [
            row
            for row in read_csv(path)
            if row["manifold"] == manifold
        ]
        if rows and "raw_slope" in rows[0]:
            raw = np.asarray([float(row["raw_slope"]) for row in rows])
            reference = np.asarray(
                [float(row["reference_optimal_sigma"]) for row in rows]
            )
            denominator = float(np.mean(reference))
            if denominator > 0.0:
                estimate = float(np.mean(raw) / denominator)
                if len(raw) > 1:
                    influence = (raw - estimate * reference) / denominator
                    ci = float(
                        student_t.ppf(
                            0.5 + 0.5 * confidence_level,
                            len(raw) - 1,
                        )
                        * np.std(influence, ddof=1)
                        / np.sqrt(len(raw))
                    )
                else:
                    ci = 0.0
                return estimate, ci
        if rows and "normalized_slope" in rows[0]:
            slopes = np.asarray(
                [float(row["normalized_slope"]) for row in rows]
            )
            return (
                float(np.mean(slopes)),
                float(
                    t_confidence_halfwidth(
                        slopes,
                        confidence_level,
                        axis=0,
                    )
                ),
            )

    centered = values - 1.0
    slope = float(
        centered @ (normalized_mean - 1.0) / (centered @ centered)
    )
    return slope, float("nan")


def plot_optimal_sigma(
    manifold: str,
    key: str,
    results_dir: Path,
    output_dir: Path,
    optima: dict[tuple[str, str], dict[str, Array]],
    normalized_rho: dict[str, dict[str, Array]],
    fit_parameters: dict[tuple[str, str], float],
    metadata: dict[str, object],
    confidence_level: float,
    figure_size: tuple[float, float],
    formats: tuple[str, ...],
) -> list[Path]:
    summary = optima[(manifold, key)]
    values = summary["values"]
    figure, axis = plt.subplots(figsize=figure_size)

    if key == "rho":
        observed = normalized_rho[manifold]
        observed_mean = observed["mean"]
        observed_ci = observed["ci"]
    else:
        observed_mean = summary["mean"]
        observed_ci = summary["ci"]

    axis.errorbar(
        values,
        observed_mean,
        yerr=observed_ci,
        color="#1f6fba",
        marker="o",
        capsize=4,
        label="Observed",
        zorder=3,
    )

    if key == "rho":
        dense = np.linspace(values[0], values[-1], 300)
        slope, slope_ci = rho_slope_and_ci(
            results_dir,
            manifold,
            values,
            observed_mean,
            confidence_level,
        )
        ci_text = rf"$\pm${slope_ci:.3f}" if np.isfinite(slope_ci) else ""
        axis.plot(
            dense,
            1.0 + slope * (dense - 1.0),
            color="#6a51a3",
            linestyle="--",
            linewidth=2.8,
            label=rf"Linear fit (slope={slope:.3f}{ci_text})",
            zorder=5,
        )
        beta = fit_parameters[(manifold, "rho_full_quadratic")]
        axis.plot(
            dense,
            dense * (1.0 + beta) / (dense**2 + beta),
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=rf"Full quadratic (fitted $\beta={beta:.3g}$)",
            zorder=5,
        )
        axis.plot(
            dense,
            dense,
            color="#c3423f",
            linestyle=":",
            linewidth=3.2,
            label=r"Dominant-term prediction (slope $1$)",
            zorder=6,
        )
        axis.set_xlabel(r"Tangential scale $\rho_\parallel$")
        axis.set_ylabel(r"$\sigma^*(\rho_\parallel)/\sigma^*(1)$")
    elif key == "lorth":
        dense = np.linspace(values[0], values[-1], 300)
        normal_dimension = metadata_argument(
            metadata,
            "reference_normal_dimension",
            80.0,
        )
        q_value = fit_parameters[(manifold, "lorth_reciprocal")]
        reference_index = int(np.argmin(np.abs(values)))
        reference_sigma = float(summary["mean"][reference_index])
        dense_descriptive = (
            reference_sigma
            * q_value
            / (normal_dimension * dense**2 + q_value)
        )
        axis.plot(
            dense,
            dense_descriptive,
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=(
                r"Descriptive fit: $a/(L_{\rm orth}^2(d-m)+q)$"
                + rf", $q={q_value:.3g}$"
            ),
            zorder=6,
        )
        valid = np.isfinite(summary["dominant"])
        axis.plot(
            values[valid],
            summary["dominant"][valid],
            color="#c3423f",
            linestyle=":",
            linewidth=3.2,
            label=r"Dominant-term prediction $\propto L_{\rm orth}^{-2}$",
            zorder=7,
        )
        axis.set_xlabel(r"Orthogonal response $L_{\rm orth}$")
        axis.set_ylabel(r"Optimal bandwidth $\sigma^*$")
    elif key == "latent_dim":
        dense = np.linspace(values[0], values[-1], 300)
        intrinsic_dimension = INTRINSIC_DIMENSION[manifold]
        lorth = metadata_argument(metadata, "dimension_lorth", 0.5)
        q_value = fit_parameters[(manifold, "dimension_reciprocal")]
        penalties = lorth**2 * (values - intrinsic_dimension)
        reference_index = int(np.argmin(penalties))
        reference_penalty = float(penalties[reference_index])
        reference_sigma = float(summary["mean"][reference_index])
        dense_penalty = lorth**2 * (dense - intrinsic_dimension)
        dense_descriptive = (
            reference_sigma
            * (reference_penalty + q_value)
            / (dense_penalty + q_value)
        )
        axis.plot(
            dense,
            dense_descriptive,
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=(
                r"Descriptive fit: $a/(L_{\rm orth}^2(d-m)+q)$"
                + rf", $q={q_value:.3g}$"
            ),
            zorder=6,
        )
        valid = np.isfinite(summary["dominant"])
        axis.plot(
            values[valid],
            summary["dominant"][valid],
            color="#c3423f",
            linestyle=":",
            linewidth=3.2,
            label=r"Dominant-term prediction $\propto(d-m)^{-1}$",
            zorder=7,
        )
        axis.set_xlabel(r"Latent dimension $d$")
        axis.set_xticks(values)
        axis.set_ylabel(r"Optimal bandwidth $\sigma^*$")
    else:
        raise KeyError(key)

    axis.grid(alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda index: 0 if labels[index] == "Observed" else 1,
    )
    axis.legend(
        [handles[index] for index in order],
        [labels[index] for index in order],
        frameon=False,
    )
    figure.tight_layout()
    return save_figure(
        figure,
        output_dir / manifold / f"{manifold}_explicit_optimal_sigma_vs_{key}",
        formats,
    )


def plot_parameter_verification(
    manifold: str,
    results_dir: Path,
    output_dir: Path,
    confidence_level: float,
    figure_size: tuple[float, float],
    formats: tuple[str, ...],
) -> list[Path]:
    rows = [
        row
        for row in read_csv(results_dir / "explicit_parameter_verification.csv")
        if row["manifold"] == manifold
    ]
    figure, axes = plt.subplots(2, 2, figsize=figure_size)
    specifications = (
        ("delta", r"Prescribed $\delta$", r"Measured $\delta$"),
        (
            "rho",
            r"Prescribed $\rho_\parallel$",
            r"Measured $\rho_\parallel$",
        ),
        (
            "lorth",
            r"Prescribed $L_{\rm orth}$",
            r"Measured $L_{\rm orth}$",
        ),
        (
            "latent_dim",
            r"Prescribed $d-m$",
            r"Measured $d-m$",
        ),
    )
    for axis, (key, xlabel, ylabel) in zip(axes.flat, specifications):
        selected = [row for row in rows if row["parameter"] == key]
        if key == "latent_dim":
            raw_x = np.asarray(
                [
                    float(row["latent_dimension"])
                    - float(row["intrinsic_dimension"])
                    for row in selected
                ]
            )
        else:
            raw_x = np.asarray(
                [float(row["prescribed_value"]) for row in selected]
            )
        raw_y = np.asarray([float(row["measured_value"]) for row in selected])
        x = np.unique(raw_x)
        y = np.asarray([np.mean(raw_y[np.isclose(raw_x, value)]) for value in x])
        yerr = np.asarray(
            [
                t_confidence_halfwidth(
                    raw_y[np.isclose(raw_x, value)],
                    confidence_level,
                    axis=0,
                )
                for value in x
            ]
        )
        bounds = [min(np.min(x), np.min(y)), max(np.max(x), np.max(y))]
        padding = 0.04 * max(1.0, bounds[1] - bounds[0])
        axis.errorbar(
            x,
            y,
            yerr=yerr,
            color="#1f6fba",
            marker="o",
            capsize=4,
            label="Finite-difference measurement",
            zorder=3,
        )
        axis.plot(
            [bounds[0] - padding, bounds[1] + padding],
            [bounds[0] - padding, bounds[1] + padding],
            color="#c3423f",
            linestyle=":",
            linewidth=2.6,
            label="Exact identity",
            zorder=6,
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    return save_figure(
        figure,
        output_dir / manifold / f"{manifold}_explicit_parameter_verification",
        formats,
    )


def plot_grid_stability(
    manifold: str,
    results_dir: Path,
    output_dir: Path,
    figure_size: tuple[float, float],
    formats: tuple[str, ...],
) -> list[Path]:
    rows = [
        row
        for row in read_csv(
            results_dir / "explicit_bandwidth_grid_stability.csv"
        )
        if row["manifold"] == manifold
        and int(row["coarse_stride"]) == 4
    ]
    colors = {
        "rho": "#1f77b4",
        "delta": "#2ca02c",
        "lorth": "#9467bd",
        "latent_dim": "#ff7f0e",
        "ambient_smoothing": "#303030",
    }
    labels = {
        "rho": r"$\rho_\parallel$",
        "delta": r"$\delta$",
        "lorth": r"$L_{\rm orth}$",
        "latent_dim": r"$d$",
        "ambient_smoothing": "Ambient",
    }
    figure, axis = plt.subplots(figsize=figure_size)
    for key, color in colors.items():
        selected = [row for row in rows if row["parameter"] == key]
        if selected:
            axis.scatter(
                [float(row["full_optimal_sigma"]) for row in selected],
                [float(row["coarse_optimal_sigma"]) for row in selected],
                color=color,
                alpha=0.75,
                label=labels[key],
                zorder=3,
            )
    maximum = max(
        max(float(row["full_optimal_sigma"]), float(row["coarse_optimal_sigma"]))
        for row in rows
    )
    axis.plot(
        [0.0, maximum],
        [0.0, maximum],
        color="#c3423f",
        linestyle=":",
        linewidth=3.0,
        label="No grid effect",
        zorder=6,
    )
    axis.set_xlabel(r"$\sigma^*$ on the full grid")
    axis.set_ylabel(r"$\sigma^*$ on the quarter-density grid")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return save_figure(
        figure,
        output_dir / manifold / f"{manifold}_bandwidth_grid_stability",
        formats,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("explicit_global_encoder_decoder_results"),
        help="Directory containing the saved CSV files and run_metadata.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figure destination. Defaults to results-dir.",
    )
    parser.add_argument(
        "--only",
        choices=("all", "w2", "optima", "diagnostics"),
        default="all",
    )
    parser.add_argument(
        "--manifolds",
        nargs="+",
        choices=tuple(MANIFOLD_LATEX),
        default=None,
    )
    parser.add_argument("--w2-width", type=float, default=11.0)
    parser.add_argument("--w2-height", type=float, default=7.0)
    parser.add_argument("--other-width", type=float, default=8.0)
    parser.add_argument("--other-height", type=float, default=7.0)
    parser.add_argument(
        "--verification-width",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--verification-height",
        type=float,
        default=8.0,
    )
    parser.add_argument(
        "--display-sigma-max",
        type=float,
        default=None,
        help="Override the displayed sigma range without changing the data.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png"),
        default=("pdf", "png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else results_dir
    )
    metadata = read_metadata(results_dir)
    confidence_level = metadata_argument(metadata, "confidence_level", 0.95)
    display_sigma_max = (
        float(args.display_sigma_max)
        if args.display_sigma_max is not None
        else metadata_argument(metadata, "display_sigma_max", 0.1)
    )
    formats = tuple(args.formats)

    summary_curves = load_summary_curves(results_dir)
    run_curves = load_run_curves(results_dir)
    optima = load_optima(results_dir)
    available_manifolds = sorted(
        {
            manifold
            for manifold, parameter in summary_curves
            if parameter != "ambient_smoothing"
        }
    )
    manifolds = args.manifolds or available_manifolds
    missing = sorted(set(manifolds) - set(available_manifolds))
    if missing:
        raise ValueError(f"No saved curves found for manifolds: {missing}")

    configure_plot_style()
    written: list[Path] = []

    if args.only in ("all", "w2"):
        for manifold in manifolds:
            for key in PARAMETER_KEYS:
                for include_ambient in (False, True):
                    written.extend(
                        plot_w2_curves(
                            manifold,
                            key,
                            summary_curves,
                            run_curves,
                            optima,
                            output_dir,
                            display_sigma_max,
                            confidence_level,
                            (args.w2_width, args.w2_height),
                            include_ambient,
                            formats,
                        )
                    )

    if args.only in ("all", "optima"):
        normalized_rho = load_normalized_rho(results_dir)
        fit_parameters = load_fit_parameters(results_dir)
        for manifold in manifolds:
            for key in ("rho", "lorth", "latent_dim"):
                written.extend(
                    plot_optimal_sigma(
                        manifold,
                        key,
                        results_dir,
                        output_dir,
                        optima,
                        normalized_rho,
                        fit_parameters,
                        metadata,
                        confidence_level,
                        (args.other_width, args.other_height),
                        formats,
                    )
                )

    if args.only in ("all", "diagnostics"):
        for manifold in manifolds:
            written.extend(
                plot_parameter_verification(
                    manifold,
                    results_dir,
                    output_dir,
                    confidence_level,
                    (args.verification_width, args.verification_height),
                    formats,
                )
            )
            written.extend(
                plot_grid_stability(
                    manifold,
                    results_dir,
                    output_dir,
                    (args.other_width, args.other_height),
                    formats,
                )
            )

    print("Plot-only regeneration completed.")
    print(f"  results read from : {results_dir}")
    print(f"  figures written to: {output_dir}")
    print(f"  manifolds         : {', '.join(manifolds)}")
    print(
        "  W2 figure size    : "
        f"({args.w2_width:g}, {args.w2_height:g})"
    )
    print(f"  files written     : {len(written)}")


if __name__ == "__main__":
    main()
