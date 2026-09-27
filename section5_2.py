#!/usr/bin/env python3
"""Experiment 5.2
Validation experiment for the latent-smoothing theorem.

This script tests latent Gaussian smoothing with one explicit encoder

    f_d : R^D -> R^d

and one explicit smooth decoder

    g : R^d -> R^D.

The ambient dimension is fixed at D=100 by default.  The encoder is coordinate
projection, and the decoder blends anchor-centered exponential-map decoders
with smooth cardinal Shepard weights.  Thus the experiment uses actual global
maps f and g, rather than directly inserting theorem parameters into a sampling
formula.  At every empirical anchor x_i, the construction satisfies

    ||g(f(x_i)) - x_i|| = delta,
    J_g(f(x_i)) J_f(x_i)|T_{x_i}M = rho_parallel I,
    ||J_g(f(x_i)) v_perp|| = L_orth ||v_perp||.

The script verifies these identities by central finite differences and then
computes W_2^2(mu_out, mu) over sigma while varying rho_parallel, delta,
L_orth, and d one at a time.  For every parameter it saves a W_2^2-versus-sigma
plot and a parameter-versus-optimal-sigma plot.  Both S^2 and S^1 x S^2 are
tested in the same fixed ambient space R^100.

Every W_2^2-versus-sigma figure also includes the ambient-smoothing reference

    W_2^2(N(0, sigma^2 I_D) * hat(mu)_n, mu).

The ambient and latent curves use the same empirical-anchor indices, target
sample, and Gaussian coordinates within each Monte Carlo repetition.
All reported discrete Wasserstein costs are computed with POT's unregularized
``ot.emd2`` solver.  The experiment does not use Sinkhorn regularization.

For each parameter family, two W_2^2-versus-sigma figures are saved: one with
the ambient reference and one containing only the latent-smoothing curves.
Both figures include a local inset of W_2^2(sigma)-W_2^2(0), together with
markers at the estimated optimal bandwidths.  Negative values in the inset
make the initial improvement visible even when it is hidden by the full-range
vertical scale.

Each Monte Carlo run draws a fresh empirical measure, a fresh independent
target discretization, and fresh Gaussian noise.  Within a run, the same
anchors, target points, mixture indices, and Gaussian coordinates are reused
for every parameter value.  The resulting comparisons therefore include
empirical-sample variability while retaining common-random-number variance
reduction within each parameter family.

The script reports 95 percent t confidence intervals, saves all run-level
curves and optima, checks sensitivity of the estimated minimizer to coarser
bandwidth grids, and flags minima that occur at a grid boundary.  Normalized
optimal-bandwidth ratios use ratios of means with paired delta-method t
intervals, so a zero reference optimum in one run does not cause division by
zero.  For
rho_parallel, L_orth, and d-m, the optimal-bandwidth figures show both an
anchored dominant-term prediction with no fitted nuisance parameter and a
clearly labelled descriptive fit containing one nuisance parameter.  The
reconstruction-error experiment
is reported only through its complete W_2^2 curves: the theorem does not give
a universal monotone formula for sigma*(delta), because delta enters both the
baseline and the bandwidth-dependent coefficients.

The codes are written together with ChatGPT
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import i0e, i1e
from scipy.stats import t as student_t

try:
    import ot
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "This experiment requires Python Optimal Transport. "
        "Install it with `pip install POT`."
    ) from error


Array = np.ndarray


# =============================================================================
# 0. Target distributions
# =============================================================================
S2_WEIGHTS = np.array([0.62, 0.38])
S2_MEANS = np.array(
    [[0.0, 0.0, 1.0], [0.88, 0.15, -0.45]], dtype=float
)
S2_MEANS /= np.linalg.norm(S2_MEANS, axis=1, keepdims=True)
S2_KAPPAS = np.array([8.0, 11.0])

S1_WEIGHTS = np.array([0.58, 0.42])
S1_MEAN_ANGLES = np.array([0.25, 3.65])
S1_KAPPAS = np.array([6.0, 9.0])


def normalize_rows(points: Array, floor: float = 1.0e-14) -> Array:
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / np.maximum(norms, floor)


def tangent_basis_s2_single(direction: Array) -> Array:
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(direction @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    first = np.cross(direction, reference)
    first /= np.linalg.norm(first)
    second = np.cross(direction, first)
    second /= np.linalg.norm(second)
    return np.column_stack([first, second])


def sample_spherical_cluster(
    rng: np.random.Generator,
    n_points: int,
    center: Array,
    concentration: float,
) -> Array:
    """Sample density proportional to exp(kappa <center,x>) on S^2."""
    center = np.asarray(center, dtype=float)
    center /= np.linalg.norm(center)
    if concentration < 1.0e-10:
        return normalize_rows(rng.normal(size=(n_points, 3)))

    uniform = rng.random(n_points)
    axial = 1.0 + np.log(
        uniform
        + (1.0 - uniform) * np.exp(-2.0 * concentration)
    ) / concentration
    axial = np.clip(axial, -1.0, 1.0)
    azimuth = 2.0 * np.pi * rng.random(n_points)
    basis = tangent_basis_s2_single(center)
    radial = np.sqrt(np.maximum(0.0, 1.0 - axial**2))
    tangent_part = radial[:, None] * (
        np.cos(azimuth)[:, None] * basis[:, 0][None, :]
        + np.sin(azimuth)[:, None] * basis[:, 1][None, :]
    )
    return normalize_rows(
        tangent_part + axial[:, None] * center[None, :]
    )


def sample_s2_base(rng: np.random.Generator, n_points: int) -> Array:
    labels = rng.choice(2, size=n_points, p=S2_WEIGHTS)
    samples = np.empty((n_points, 3), dtype=float)
    for component in range(2):
        locations = np.flatnonzero(labels == component)
        if locations.size:
            samples[locations] = sample_spherical_cluster(
                rng,
                locations.size,
                S2_MEANS[component],
                float(S2_KAPPAS[component]),
            )
    return samples


def sample_s1_base(rng: np.random.Generator, n_points: int) -> Array:
    labels = rng.choice(2, size=n_points, p=S1_WEIGHTS)
    angles = np.empty(n_points, dtype=float)
    for component in range(2):
        locations = np.flatnonzero(labels == component)
        if locations.size:
            angles[locations] = rng.vonmises(
                S1_MEAN_ANGLES[component],
                S1_KAPPAS[component],
                size=locations.size,
            )
    return np.column_stack([np.cos(angles), np.sin(angles)])


def sample_product_base(rng: np.random.Generator, n_points: int) -> Array:
    return np.column_stack(
        [sample_s1_base(rng, n_points), sample_s2_base(rng, n_points)]
    )


def s2_population_mean_base() -> Array:
    resultants = 1.0 / np.tanh(S2_KAPPAS) - 1.0 / S2_KAPPAS
    return np.sum(
        S2_WEIGHTS[:, None]
        * resultants[:, None]
        * S2_MEANS,
        axis=0,
    )


def product_population_mean_base() -> Array:
    resultants = i1e(S1_KAPPAS) / i0e(S1_KAPPAS)
    directions = np.column_stack(
        [np.cos(S1_MEAN_ANGLES), np.sin(S1_MEAN_ANGLES)]
    )
    circle_mean = np.sum(
        S1_WEIGHTS[:, None]
        * resultants[:, None]
        * directions,
        axis=0,
    )
    return np.concatenate([circle_mean, s2_population_mean_base()])


# =============================================================================
# 1. Fixed-ambient manifold geometry
# =============================================================================
def pad_to_ambient(points: Array, ambient_dimension: int) -> Array:
    result = np.zeros((len(points), ambient_dimension), dtype=float)
    result[:, : points.shape[1]] = points
    return result


def sphere_tangent_frames(points: Array, ambient_dimension: int) -> Array:
    frames = np.zeros((len(points), ambient_dimension, 2), dtype=float)
    for index, point in enumerate(points[:, :3]):
        frames[index, :3, :] = tangent_basis_s2_single(point)
    return frames


def product_tangent_frames(points: Array, ambient_dimension: int) -> Array:
    frames = np.zeros((len(points), ambient_dimension, 3), dtype=float)
    circle = points[:, :2]
    sphere = points[:, 2:5]
    frames[:, :2, 0] = np.column_stack([-circle[:, 1], circle[:, 0]])
    for index, point in enumerate(sphere):
        frames[index, 2:5, 1:] = tangent_basis_s2_single(point)
    return frames


def sphere_exponential_map(base_points: Array, tangent_vectors: Array) -> Array:
    norms = np.linalg.norm(tangent_vectors, axis=1, keepdims=True)
    factors = np.ones_like(norms)
    nonzero = norms[:, 0] > 1.0e-14
    factors[nonzero] = np.sin(norms[nonzero]) / norms[nonzero]
    return np.cos(norms) * base_points + factors * tangent_vectors


def product_exponential_map(base_points: Array, tangent_vectors: Array) -> Array:
    result = np.zeros_like(base_points)
    result[:, :2] = sphere_exponential_map(
        base_points[:, :2], tangent_vectors[:, :2]
    )
    result[:, 2:5] = sphere_exponential_map(
        base_points[:, 2:5], tangent_vectors[:, 2:5]
    )
    return result


@dataclass(frozen=True)
class ManifoldSpec:
    key: str
    latex_name: str
    intrinsic_dimension: int
    natural_ambient_dimension: int
    sample_base: Callable[[np.random.Generator, int], Array]
    population_mean_base: Callable[[], Array]
    tangent_frames: Callable[[Array, int], Array]
    exponential_map: Callable[[Array, Array], Array]


MANIFOLDS = (
    ManifoldSpec(
        key="sphere",
        latex_name=r"$\mathbb S^2$",
        intrinsic_dimension=2,
        natural_ambient_dimension=3,
        sample_base=sample_s2_base,
        population_mean_base=s2_population_mean_base,
        tangent_frames=sphere_tangent_frames,
        exponential_map=sphere_exponential_map,
    ),
    ManifoldSpec(
        key="product",
        latex_name=r"$\mathbb S^1\times\mathbb S^2$",
        intrinsic_dimension=3,
        natural_ambient_dimension=5,
        sample_base=sample_product_base,
        population_mean_base=product_population_mean_base,
        tangent_frames=product_tangent_frames,
        exponential_map=product_exponential_map,
    ),
)


# =============================================================================
# 2. Explicit encoder and decoder
# =============================================================================
@dataclass
class ExplicitEncoderDecoder:
    spec: ManifoldSpec
    anchors: Array
    ambient_dimension: int
    latent_dimension: int
    rho_parallel: float
    delta: float
    l_orth: float
    bias_direction: Array
    shepard_power: int = 2
    decode_chunk_size: int = 256

    def __post_init__(self) -> None:
        d0 = self.spec.natural_ambient_dimension
        if not d0 <= self.latent_dimension <= self.ambient_dimension:
            raise ValueError(
                "Require natural ambient dimension <= d <= D."
            )
        if self.anchors.shape[1] != self.ambient_dimension:
            raise ValueError("Anchor ambient dimension is inconsistent.")
        self.centers = self.anchors[:, : self.latent_dimension].copy()
        self.frames = self.spec.tangent_frames(
            self.anchors,
            self.ambient_dimension,
        )
        self.latent_frames = self.frames[:, : self.latent_dimension, :]
        gram = np.einsum(
            "ndm,ndk->nmk",
            self.latent_frames,
            self.latent_frames,
        )
        target = np.eye(self.spec.intrinsic_dimension)[None, :, :]
        if float(np.max(np.abs(gram - target))) > 1.0e-10:
            raise ValueError("The projected tangent frame is not isometric.")

    def encode(self, points: Array) -> Array:
        return np.asarray(points, dtype=float)[:, : self.latent_dimension]

    def _weights(self, latent_points: Array) -> Array:
        d0 = self.spec.natural_ambient_dimension
        locations = latent_points[:, :d0]
        anchor_locations = self.centers[:, :d0]
        differences = locations[:, None, :] - anchor_locations[None, :, :]
        distances_squared = np.sum(differences**2, axis=2)
        weights = np.zeros_like(distances_squared)

        nearest = np.argmin(distances_squared, axis=1)
        minimum = distances_squared[np.arange(len(latent_points)), nearest]
        exact = minimum < 1.0e-24
        if np.any(exact):
            rows = np.flatnonzero(exact)
            weights[rows, nearest[rows]] = 1.0

        regular = ~exact
        if np.any(regular):
            log_weights = -float(self.shepard_power) * np.log(
                distances_squared[regular]
            )
            log_weights -= np.max(log_weights, axis=1, keepdims=True)
            stable = np.exp(log_weights)
            weights[regular] = stable / np.sum(
                stable, axis=1, keepdims=True
            )
        return weights

    def _decode_chunk(self, latent_points: Array) -> Array:
        batch_size = len(latent_points)
        anchor_count = len(self.anchors)
        d = self.latent_dimension
        D = self.ambient_dimension

        weights = self._weights(latent_points)
        displacement = (
            latent_points[:, None, :] - self.centers[None, :, :]
        )
        tangent_coordinates = np.einsum(
            "ndm,bnd->bnm",
            self.latent_frames,
            displacement,
        )
        ambient_tangent = np.einsum(
            "nDm,bnm->bnD",
            self.frames,
            tangent_coordinates,
        )
        latent_tangent = np.einsum(
            "ndm,bnm->bnd",
            self.latent_frames,
            tangent_coordinates,
        )
        latent_orthogonal = displacement - latent_tangent
        ambient_orthogonal = np.zeros(
            (batch_size, anchor_count, D), dtype=float
        )
        ambient_orthogonal[:, :, :d] = latent_orthogonal

        repeated_anchors = np.broadcast_to(
            self.anchors[None, :, :],
            (batch_size, anchor_count, D),
        ).reshape(-1, D)
        exponential = self.spec.exponential_map(
            repeated_anchors,
            (self.rho_parallel * ambient_tangent).reshape(-1, D),
        ).reshape(batch_size, anchor_count, D)

        local_outputs = exponential + self.l_orth * ambient_orthogonal
        blended = np.einsum("bn,bnD->bD", weights, local_outputs)
        return blended + self.delta * self.bias_direction[None, :]

    def decode(self, latent_points: Array) -> Array:
        latent_points = np.asarray(latent_points, dtype=float)
        result = np.empty(
            (len(latent_points), self.ambient_dimension), dtype=float
        )
        for start in range(0, len(latent_points), self.decode_chunk_size):
            stop = min(start + self.decode_chunk_size, len(latent_points))
            result[start:stop] = self._decode_chunk(
                latent_points[start:stop]
            )
        return result


# =============================================================================
# 3. Diagnostics
# =============================================================================
def finite_difference_jacobians(
    model: ExplicitEncoderDecoder,
    anchor_indices: Array,
    step: float,
) -> Array:
    centers = model.centers[anchor_indices]
    d = model.latent_dimension
    identity = np.eye(d)
    plus = centers[:, None, :] + step * identity[None, :, :]
    minus = centers[:, None, :] - step * identity[None, :, :]
    plus_values = model.decode(plus.reshape(-1, d)).reshape(
        len(centers), d, model.ambient_dimension
    )
    minus_values = model.decode(minus.reshape(-1, d)).reshape(
        len(centers), d, model.ambient_dimension
    )
    derivatives = (plus_values - minus_values) / (2.0 * step)
    return np.transpose(derivatives, (0, 2, 1))


def verify_model(
    model: ExplicitEncoderDecoder,
    finite_difference_step: float,
    maximum_checked_anchors: int,
) -> dict[str, float]:
    reconstructed = model.decode(model.centers)
    reconstruction_distances = np.linalg.norm(
        reconstructed - model.anchors,
        axis=1,
    )
    rms_reconstruction = float(
        np.sqrt(np.mean(reconstruction_distances**2))
    )

    checked_count = min(maximum_checked_anchors, len(model.anchors))
    checked = np.linspace(
        0,
        len(model.anchors) - 1,
        checked_count,
        dtype=int,
    )
    jacobians = finite_difference_jacobians(
        model,
        checked,
        finite_difference_step,
    )
    frames = model.frames[checked]
    latent_frames = model.latent_frames[checked]
    tangent_maps = np.einsum(
        "nDd,ndm->nDm",
        jacobians,
        latent_frames,
    )
    expected_tangent = model.rho_parallel * frames

    tangent_errors = np.asarray(
        [
            np.linalg.norm(
                tangent_maps[index] - expected_tangent[index],
                ord=2,
            )
            for index in range(checked_count)
        ]
    )
    measured_rho = np.einsum(
        "nDm,nDm->n",
        tangent_maps,
        frames,
    ) / model.spec.intrinsic_dimension

    orthogonal_responses: list[float] = []
    orthogonal_errors: list[float] = []
    complement_dimensions: list[int] = []
    for local_index in range(checked_count):
        latent_frame = latent_frames[local_index]
        _, _, right = np.linalg.svd(
            latent_frame.T,
            full_matrices=True,
        )
        complement = right[model.spec.intrinsic_dimension :].T
        complement_dimensions.append(complement.shape[1])
        if complement.shape[1] == 0:
            response = 0.0
            error = abs(model.l_orth)
        else:
            mapped = jacobians[local_index] @ complement
            singular_values = np.linalg.svd(
                mapped,
                compute_uv=False,
            )
            response = float(np.max(singular_values))
            expected = model.l_orth * np.pad(
                complement,
                (
                    (0, model.ambient_dimension - model.latent_dimension),
                    (0, 0),
                ),
            )
            error = float(np.linalg.norm(mapped - expected, ord=2))
        orthogonal_responses.append(response)
        orthogonal_errors.append(error)

    gram = np.einsum(
        "ndm,ndk->nmk",
        model.latent_frames,
        model.latent_frames,
    )
    gram_target = np.eye(model.spec.intrinsic_dimension)[None, :, :]
    gram_errors = np.asarray(
        [
            np.linalg.norm(matrix, ord=2)
            for matrix in (gram - gram_target)
        ]
    )

    return {
        "rms_reconstruction": rms_reconstruction,
        "reconstruction_parameter_error": abs(
            rms_reconstruction - model.delta
        ),
        "rho_measured_mean": float(np.mean(measured_rho)),
        "rho_measured_std": float(np.std(measured_rho)),
        "tangent_map_max_error": float(np.max(tangent_errors)),
        "lorth_measured_mean": float(np.mean(orthogonal_responses)),
        "lorth_measured_std": float(np.std(orthogonal_responses)),
        "orthogonal_map_max_error": float(np.max(orthogonal_errors)),
        "encoder_gram_max_error": float(np.max(gram_errors)),
        "complement_dimension_min": float(min(complement_dimensions)),
        "complement_dimension_max": float(max(complement_dimensions)),
    }


# =============================================================================
# 4. Wasserstein distance and curve summaries
# =============================================================================
def discrete_w2_squared(first: Array, second: Array) -> float:
    """Compute exact discrete Euclidean W_2^2 using POT's network simplex."""
    cost = ot.dist(
        np.asarray(first, dtype=np.float64),
        np.asarray(second, dtype=np.float64),
        metric="sqeuclidean",
    )
    return exact_uniform_transport_cost(cost)


def exact_uniform_transport_cost(cost: Array) -> float:
    """Return the unregularized OT cost between two uniform measures.

    POT's ``ot.emd2`` solves the discrete Kantorovich problem by network
    simplex. No entropic regularization or Sinkhorn approximation is used.
    """
    cost = np.ascontiguousarray(cost, dtype=np.float64)
    n_first, n_second = cost.shape
    first_weights = np.full(n_first, 1.0 / n_first, dtype=np.float64)
    second_weights = np.full(n_second, 1.0 / n_second, dtype=np.float64)
    return float(
        ot.emd2(
            first_weights,
            second_weights,
            cost,
            numItermax=1_000_000,
        )
    )


def intrinsic_squared_cost(
    spec: ManifoldSpec,
    first: Array,
    second: Array,
) -> Array:
    """Return the pairwise squared intrinsic distance on the test manifold."""
    if spec.key == "sphere":
        dot = np.clip(first[:, :3] @ second[:, :3].T, -1.0, 1.0)
        return np.arccos(dot) ** 2
    if spec.key == "product":
        circle_dot = np.clip(
            first[:, :2] @ second[:, :2].T,
            -1.0,
            1.0,
        )
        sphere_dot = np.clip(
            first[:, 2:5] @ second[:, 2:5].T,
            -1.0,
            1.0,
        )
        return np.arccos(circle_dot) ** 2 + np.arccos(sphere_dot) ** 2
    raise KeyError(f"Unknown manifold: {spec.key}")


def discrete_intrinsic_w2_squared(
    spec: ManifoldSpec,
    first: Array,
    second: Array,
) -> float:
    """Compute exact discrete intrinsic W_2^2 with POT's ``ot.emd2``."""
    cost = intrinsic_squared_cost(spec, first, second)
    return exact_uniform_transport_cost(cost)


def mixture_anchor_indices(
    n_samples: int,
    n_anchors: int,
    rng: np.random.Generator,
) -> Array:
    """Draw iid indices from the uniform empirical mixing measure."""
    return rng.integers(0, n_anchors, size=n_samples)


def refined_grid_minimum(x_values: Array, y_values: Array) -> tuple[float, float]:
    index = int(np.argmin(y_values))
    if index == 0 or index == len(x_values) - 1:
        return float(x_values[index]), float(y_values[index])
    local_x = x_values[index - 1 : index + 2]
    local_y = y_values[index - 1 : index + 2]
    quadratic = np.polyfit(local_x, local_y, deg=2)
    curvature, linear, _ = quadratic
    if curvature <= 0.0:
        return float(x_values[index]), float(y_values[index])
    vertex = float(-linear / (2.0 * curvature))
    if not local_x[0] <= vertex <= local_x[-1]:
        return float(x_values[index]), float(y_values[index])
    return vertex, float(np.polyval(quadratic, vertex))


def t_confidence_halfwidth(
    samples: Array,
    confidence_level: float,
    axis: int = 0,
) -> Array:
    """Two-sided Student-t confidence half-width for the sample mean."""
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


def summarize_curves(
    sigma_grid: Array,
    curves: Array,
    confidence_level: float,
) -> dict[str, Array]:
    mean_curve = np.mean(curves, axis=0)
    std_curve = (
        np.std(curves, axis=0, ddof=1)
        if curves.shape[0] > 1
        else np.zeros_like(mean_curve)
    )
    repetitions, parameter_count, _ = curves.shape
    optima = np.empty((repetitions, parameter_count), dtype=float)
    minima = np.empty_like(optima)
    for repetition in range(repetitions):
        for parameter_index in range(parameter_count):
            optima[repetition, parameter_index], minima[
                repetition, parameter_index
            ] = refined_grid_minimum(
                sigma_grid,
                curves[repetition, parameter_index],
            )
    return {
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "curve_ci_halfwidth": t_confidence_halfwidth(
            curves,
            confidence_level,
            axis=0,
        ),
        "optima": optima,
        "minima": minima,
        "mean_optimum": np.mean(optima, axis=0),
        "std_optimum": (
            np.std(optima, axis=0, ddof=1)
            if repetitions > 1
            else np.zeros(parameter_count)
        ),
        "optimum_ci_halfwidth": t_confidence_halfwidth(
            optima,
            confidence_level,
            axis=0,
        ),
        "mean_minimum": np.mean(minima, axis=0),
        "std_minimum": (
            np.std(minima, axis=0, ddof=1)
            if repetitions > 1
            else np.zeros(parameter_count)
        ),
        "minimum_ci_halfwidth": t_confidence_halfwidth(
            minima,
            confidence_level,
            axis=0,
        ),
    }


def normalized_optimum_summary(
    summary: dict[str, Array],
    reference_index: int,
    confidence_level: float,
) -> dict[str, Array]:
    """Estimate ratios of mean optima relative to one reference setting.

    The target quantity is

        E[sigma*(parameter)] / E[sigma*(reference)].

    We estimate it by the corresponding ratio of sample means.  This is
    preferable to averaging run-wise ratios because an individual Monte
    Carlo run can have a boundary minimizer sigma*=0.  Such a run contains
    valid information but cannot appear in the denominator of a run-wise
    ratio.  The uncertainty calculation uses the paired first-order
    (delta-method) influence values X-ratio*Y and a Student-t multiplier.
    """
    optima = np.asarray(summary["optima"], dtype=float)
    reference = optima[:, reference_index]
    reference_mean = float(np.mean(reference))
    numerical_floor = 100.0 * np.finfo(float).eps
    if reference_mean <= numerical_floor:
        raise ValueError(
            "The mean reference optimal bandwidth is zero. The normalized "
            "comparison is not identifiable. Inspect the boundary-minimum "
            "flags, increase sigma resolution near zero, or reduce the "
            "fixed orthogonal penalty."
        )

    mean_optima = np.mean(optima, axis=0)
    ratios = mean_optima / reference_mean
    sample_count = len(reference)
    if sample_count > 1:
        influence = (
            optima - ratios[None, :] * reference[:, None]
        ) / reference_mean
        ratio_std = np.std(influence, axis=0, ddof=1)
        ratio_ci = (
            float(
                student_t.ppf(
                    0.5 + 0.5 * confidence_level,
                    sample_count - 1,
                )
            )
            * ratio_std
            / np.sqrt(sample_count)
        )
    else:
        influence = np.zeros_like(optima)
        ratio_std = np.zeros(optima.shape[1], dtype=float)
        ratio_ci = np.zeros(optima.shape[1], dtype=float)

    return {
        "mean": ratios,
        "std": ratio_std,
        "ci_halfwidth": ratio_ci,
        "influence": influence,
        "reference_mean": reference_mean,
        "zero_reference_run_count": int(np.sum(reference <= numerical_floor)),
        "method": "ratio_of_means_paired_delta_method",
    }


def paired_ratio_of_means(
    numerator: Array,
    denominator: Array,
    confidence_level: float,
) -> tuple[float, float, float]:
    """Ratio of paired sample means with a delta-method t interval."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("Paired ratio inputs must be one-dimensional peers.")
    denominator_mean = float(np.mean(denominator))
    numerical_floor = 100.0 * np.finfo(float).eps
    if denominator_mean <= numerical_floor:
        raise ValueError("The mean denominator of the paired ratio is zero.")
    estimate = float(np.mean(numerator) / denominator_mean)
    if len(numerator) <= 1:
        return estimate, 0.0, 0.0
    influence = (numerator - estimate * denominator) / denominator_mean
    standard_deviation = float(np.std(influence, ddof=1))
    quantile = float(
        student_t.ppf(
            0.5 + 0.5 * confidence_level,
            len(numerator) - 1,
        )
    )
    halfwidth = quantile * standard_deviation / np.sqrt(len(numerator))
    return estimate, standard_deviation, float(halfwidth)


def reciprocal_penalty_prediction(
    penalties: Array,
    observed_optima: Array,
    reference_index: int,
) -> tuple[float, Array]:
    """Fit sigma*(p) = sigma_ref (p_ref + q) / (p + q), q >= 0."""
    penalties = np.asarray(penalties, dtype=float)
    observed_optima = np.asarray(observed_optima, dtype=float)
    reference_penalty = float(penalties[reference_index])
    reference_sigma = float(observed_optima[reference_index])

    def prediction(q_value: float) -> Array:
        return (
            reference_sigma
            * (reference_penalty + q_value)
            / (penalties + q_value)
        )

    fit = minimize_scalar(
        lambda log_q: float(
            np.sum(
                (
                    observed_optima
                    - prediction(float(np.exp(log_q)))
                )
                ** 2
            )
        ),
        bounds=(-20.0, 20.0),
        method="bounded",
        options={"xatol": 1.0e-12},
    )
    if not fit.success:
        raise RuntimeError("The reciprocal penalty fit failed.")
    q_value = float(np.exp(fit.x))
    return q_value, prediction(q_value)


def rho_quadratic_prediction(
    rho_values: Array,
    observed_optima: Array,
    supplied_beta: float | None,
) -> tuple[float, Array, bool]:
    """Predict sigma*(rho) using the full quadratic denominator.

    If beta = B_n/A_n and sigma_one = sigma*(1), then the theorem's
    quadratic minimizer has the normalized form

        sigma*(rho)
          = sigma_one * rho * (1 + beta) / (rho**2 + beta).

    The theorem constants entering A_n and B_n are generally unknown in the
    numerical example.  Consequently, beta is either supplied by the user or
    fitted under the theorem-compatible constraint beta >= 0.
    """
    rho_values = np.asarray(rho_values, dtype=float)
    observed_optima = np.asarray(observed_optima, dtype=float)
    reference_index = int(np.argmin(np.abs(rho_values - 1.0)))
    sigma_one = float(observed_optima[reference_index])

    def prediction(beta: float) -> Array:
        return (
            sigma_one
            * rho_values
            * (1.0 + beta)
            / (rho_values**2 + beta)
        )

    fitted = supplied_beta is None
    if fitted:
        fit = minimize_scalar(
            lambda beta: float(
                np.sum((observed_optima - prediction(beta)) ** 2)
            ),
            bounds=(0.0, 100.0),
            method="bounded",
            options={"xatol": 1.0e-12},
        )
        if not fit.success:
            raise RuntimeError("The nonnegative rho beta fit failed.")
        beta = float(fit.x)
    else:
        beta = float(supplied_beta)

    return beta, prediction(beta), fitted


def regression_diagnostics(observed: Array, predicted: Array) -> dict[str, float]:
    """Return transparent descriptive-fit diagnostics."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = observed - predicted
    sum_squared_error = float(residual @ residual)
    centered = observed - float(np.mean(observed))
    total_sum_squares = float(centered @ centered)
    r_squared = (
        1.0 - sum_squared_error / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r_squared": r_squared,
        "max_absolute_error": float(np.max(np.abs(residual))),
    }


def grid_stability_rows(
    manifold: str,
    parameter: str,
    values: Array,
    sigma_grid: Array,
    curves: Array,
    strides: tuple[int, ...] = (2, 4),
) -> list[dict[str, object]]:
    """Compare refined optima on the full grid and nested coarser grids."""
    rows: list[dict[str, object]] = []
    repetitions, parameter_count, grid_size = curves.shape
    for repetition in range(repetitions):
        for parameter_index in range(parameter_count):
            full_sigma, _ = refined_grid_minimum(
                sigma_grid,
                curves[repetition, parameter_index],
            )
            grid_min_index = int(
                np.argmin(curves[repetition, parameter_index])
            )
            boundary = grid_min_index in (0, grid_size - 1)
            for stride in strides:
                selected = np.arange(0, grid_size, stride, dtype=int)
                if selected[-1] != grid_size - 1:
                    selected = np.append(selected, grid_size - 1)
                coarse_sigma, _ = refined_grid_minimum(
                    sigma_grid[selected],
                    curves[repetition, parameter_index, selected],
                )
                absolute_change = abs(coarse_sigma - full_sigma)
                relative_change = (
                    absolute_change / full_sigma
                    if full_sigma > 0.0
                    else float("nan")
                )
                rows.append(
                    {
                        "manifold": manifold,
                        "parameter": parameter,
                        "parameter_value": float(values[parameter_index]),
                        "repetition": repetition,
                        "coarse_stride": stride,
                        "full_grid_points": grid_size,
                        "coarse_grid_points": len(selected),
                        "full_optimal_sigma": full_sigma,
                        "coarse_optimal_sigma": coarse_sigma,
                        "absolute_change": absolute_change,
                        "relative_change": relative_change,
                        "full_grid_boundary_minimum": boundary,
                    }
                )
    return rows


# =============================================================================
# 5. One-manifold experiment
# =============================================================================
def make_model(
    spec: ManifoldSpec,
    anchors: Array,
    args: argparse.Namespace,
    latent_dimension: int,
    rho_parallel: float,
    delta: float,
    l_orth: float,
    bias_direction: Array,
) -> ExplicitEncoderDecoder:
    return ExplicitEncoderDecoder(
        spec=spec,
        anchors=anchors,
        ambient_dimension=args.ambient_dimension,
        latent_dimension=latent_dimension,
        rho_parallel=rho_parallel,
        delta=delta,
        l_orth=l_orth,
        bias_direction=bias_direction,
        shepard_power=args.shepard_power,
        decode_chunk_size=args.decode_chunk_size,
    )


def run_verification(
    spec: ManifoldSpec,
    anchors: Array,
    args: argparse.Namespace,
    bias_direction: Array,
    reference_dimension: int,
    latent_dimensions: Array,
    repetition: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    configurations: list[tuple[str, float, int, float, float, float]] = []
    for value in args.delta_values:
        configurations.append(
            (
                "delta",
                value,
                reference_dimension,
                1.0,
                value,
                args.slope_study_lorth,
            )
        )
    for value in args.rho_values:
        configurations.append(
            (
                "rho",
                value,
                reference_dimension,
                value,
                0.0,
                args.slope_study_lorth,
            )
        )
    for value in args.lorth_values:
        configurations.append(
            ("lorth", value, reference_dimension, 1.0, 0.0, value)
        )
    for value in latent_dimensions:
        configurations.append(
            (
                "latent_dim",
                float(value - spec.intrinsic_dimension),
                int(value),
                1.0,
                0.0,
                args.dimension_lorth,
            )
        )

    for parameter, value, d, rho, delta, lorth in configurations:
        model = make_model(
            spec,
            anchors,
            args,
            d,
            rho,
            delta,
            lorth,
            bias_direction,
        )
        diagnostics = verify_model(
            model,
            args.finite_difference_step,
            args.checked_anchors,
        )
        if parameter == "delta":
            measured = diagnostics["rms_reconstruction"]
        elif parameter == "rho":
            measured = diagnostics["rho_measured_mean"]
        elif parameter == "lorth":
            measured = diagnostics["lorth_measured_mean"]
        else:
            measured = diagnostics["complement_dimension_min"]
        rows.append(
            {
                "manifold": spec.key,
                "repetition": repetition,
                "parameter": parameter,
                "prescribed_value": float(value),
                "measured_value": float(measured),
                "ambient_dimension": args.ambient_dimension,
                "intrinsic_dimension": spec.intrinsic_dimension,
                "latent_dimension": d,
                "rho_parallel": rho,
                "delta": delta,
                "l_orth": lorth,
                **diagnostics,
            }
        )
    return rows


def run_manifold_experiment(
    spec: ManifoldSpec,
    args: argparse.Namespace,
) -> dict[str, object]:
    population_mean = pad_to_ambient(
        spec.population_mean_base()[None, :],
        args.ambient_dimension,
    )[0]
    bias_direction = -population_mean / np.linalg.norm(population_mean)

    normal_dimensions = np.asarray(args.normal_dimensions, dtype=int)
    latent_dimensions = spec.intrinsic_dimension + normal_dimensions
    reference_dimension = spec.intrinsic_dimension + args.reference_normal_dimension
    if reference_dimension < spec.natural_ambient_dimension:
        raise ValueError("Reference d must retain the natural embedding.")
    if np.any(latent_dimensions < spec.natural_ambient_dimension):
        raise ValueError(
            f"All d values for {spec.key} must be at least "
            f"{spec.natural_ambient_dimension}."
        )

    sigma_grid = args.sigma_max * np.linspace(
        0.0,
        1.0,
        args.sigma_points,
    ) ** args.sigma_grid_power
    rho_values = np.asarray(args.rho_values, dtype=float)
    delta_values = np.asarray(args.delta_values, dtype=float)
    lorth_values = np.asarray(args.lorth_values, dtype=float)
    shapes = {
        "rho": (args.repetitions, len(rho_values), len(sigma_grid)),
        "delta": (args.repetitions, len(delta_values), len(sigma_grid)),
        "lorth": (args.repetitions, len(lorth_values), len(sigma_grid)),
        "latent_dim": (
            args.repetitions,
            len(latent_dimensions),
            len(sigma_grid),
        ),
    }
    curves = {
        key: np.empty(shape, dtype=float) for key, shape in shapes.items()
    }
    ambient_curves = np.empty(
        (args.repetitions, len(sigma_grid)),
        dtype=float,
    )
    w0_estimates = np.empty(args.repetitions, dtype=float)
    verification_rows: list[dict[str, object]] = []
    rho_one_index = int(np.argmin(np.abs(rho_values - 1.0)))

    for repetition in range(args.repetitions):
        print(
            f"[{spec.key}] repetition {repetition + 1}/"
            f"{args.repetitions}",
            flush=True,
        )
        repetition_rng = np.random.default_rng(
            args.seed
            + 10_000 * spec.natural_ambient_dimension
            + repetition
        )
        anchors_base = spec.sample_base(
            repetition_rng,
            args.n_empirical,
        )
        anchors = pad_to_ambient(
            anchors_base,
            args.ambient_dimension,
        )
        target_base = spec.sample_base(repetition_rng, args.n_ot)
        target = pad_to_ambient(
            target_base,
            args.ambient_dimension,
        )
        verification_rows.extend(
            run_verification(
                spec,
                anchors,
                args,
                bias_direction,
                reference_dimension,
                latent_dimensions,
                repetition,
            )
        )

        rho_models = [
            make_model(
                spec,
                anchors,
                args,
                reference_dimension,
                value,
                0.0,
                args.slope_study_lorth,
                bias_direction,
            )
            for value in rho_values
        ]
        baseline_model = rho_models[rho_one_index]
        lorth_models = [
            make_model(
                spec,
                anchors,
                args,
                reference_dimension,
                1.0,
                0.0,
                value,
                bias_direction,
            )
            for value in lorth_values
        ]
        dimension_models = [
            make_model(
                spec,
                anchors,
                args,
                int(value),
                1.0,
                0.0,
                args.dimension_lorth,
                bias_direction,
            )
            for value in latent_dimensions
        ]

        w0_estimates[repetition] = np.sqrt(
            discrete_intrinsic_w2_squared(
                spec,
                anchors_base,
                target_base,
            )
        )
        anchor_indices = mixture_anchor_indices(
            args.n_ot,
            args.n_empirical,
            repetition_rng,
        )
        standard_noise = repetition_rng.normal(
            size=(args.n_ot, args.ambient_dimension)
        )

        for sigma_index, sigma in enumerate(sigma_grid):
            ambient_generated = (
                anchors[anchor_indices]
                + sigma * standard_noise
            )
            ambient_curves[repetition, sigma_index] = discrete_w2_squared(
                ambient_generated,
                target,
            )

            reference_latent = (
                baseline_model.centers[anchor_indices]
                + sigma * standard_noise[:, :reference_dimension]
            )

            rho_outputs: list[Array] = []
            for parameter_index, model in enumerate(rho_models):
                generated = model.decode(reference_latent)
                rho_outputs.append(generated)
                curves["rho"][
                    repetition, parameter_index, sigma_index
                ] = discrete_w2_squared(generated, target)

            baseline_output = rho_outputs[rho_one_index]
            for parameter_index, delta in enumerate(delta_values):
                generated = (
                    baseline_output
                    + delta * bias_direction[None, :]
                )
                curves["delta"][
                    repetition, parameter_index, sigma_index
                ] = discrete_w2_squared(generated, target)

            for parameter_index, model in enumerate(lorth_models):
                generated = model.decode(reference_latent)
                curves["lorth"][
                    repetition, parameter_index, sigma_index
                ] = discrete_w2_squared(generated, target)

            for parameter_index, model in enumerate(dimension_models):
                d = model.latent_dimension
                latent = (
                    model.centers[anchor_indices]
                    + sigma * standard_noise[:, :d]
                )
                generated = model.decode(latent)
                curves["latent_dim"][
                    repetition, parameter_index, sigma_index
                ] = discrete_w2_squared(generated, target)

    experiments = {
        "rho": {
            "values": rho_values,
            "curves": curves["rho"],
            "summary": summarize_curves(
                sigma_grid,
                curves["rho"],
                args.confidence_level,
            ),
        },
        "delta": {
            "values": delta_values,
            "curves": curves["delta"],
            "summary": summarize_curves(
                sigma_grid,
                curves["delta"],
                args.confidence_level,
            ),
        },
        "lorth": {
            "values": lorth_values,
            "curves": curves["lorth"],
            "summary": summarize_curves(
                sigma_grid,
                curves["lorth"],
                args.confidence_level,
            ),
        },
        "latent_dim": {
            "values": latent_dimensions,
            "normal_dimensions": normal_dimensions,
            "curves": curves["latent_dim"],
            "summary": summarize_curves(
                sigma_grid,
                curves["latent_dim"],
                args.confidence_level,
            ),
        },
    }
    ambient_summary = summarize_curves(
        sigma_grid,
        ambient_curves[:, None, :],
        args.confidence_level,
    )

    lorth_zero_index = int(np.argmin(np.abs(lorth_values)))
    rho_beta, rho_prediction, rho_beta_fitted = rho_quadratic_prediction(
        rho_values,
        experiments["rho"]["summary"]["mean_optimum"],
        args.rho_beta,
    )
    lorth_penalties = args.reference_normal_dimension * lorth_values**2
    lorth_q, lorth_prediction = reciprocal_penalty_prediction(
        lorth_penalties,
        experiments["lorth"]["summary"]["mean_optimum"],
        lorth_zero_index,
    )
    dimension_penalties = args.dimension_lorth**2 * normal_dimensions
    dimension_reference_index = int(np.argmin(dimension_penalties))
    dimension_q, dimension_prediction = reciprocal_penalty_prediction(
        dimension_penalties,
        experiments["latent_dim"]["summary"]["mean_optimum"],
        dimension_reference_index,
    )

    rho_normalized = normalized_optimum_summary(
        experiments["rho"]["summary"],
        rho_one_index,
        args.confidence_level,
    )
    rho_optima_by_run = experiments["rho"]["summary"]["optima"]
    rho_reference_by_run = rho_optima_by_run[:, rho_one_index]
    centered_rho = rho_values - 1.0
    rho_denominator = float(centered_rho @ centered_rho)
    rho_raw_slopes = (
        (
            rho_optima_by_run
            - rho_reference_by_run[:, None]
        )
        @ centered_rho
        / rho_denominator
    )
    (
        rho_empirical_slope,
        rho_slope_std,
        rho_slope_ci,
    ) = paired_ratio_of_means(
        rho_raw_slopes,
        rho_reference_by_run,
        args.confidence_level,
    )
    w0_mean = float(np.mean(w0_estimates))
    w0_std = (
        float(np.std(w0_estimates, ddof=1))
        if len(w0_estimates) > 1
        else 0.0
    )
    rho_prediction_normalized = (
        rho_prediction / rho_prediction[rho_one_index]
    )

    positive_lorth = np.flatnonzero(lorth_values > 0.0)
    lorth_dominant_prediction = np.full_like(lorth_values, np.nan)
    if positive_lorth.size:
        lorth_reference_index = int(positive_lorth[-1])
        lorth_reference = float(lorth_values[lorth_reference_index])
        lorth_reference_sigma = float(
            experiments["lorth"]["summary"]["mean_optimum"]
            [lorth_reference_index]
        )
        lorth_dominant_prediction[positive_lorth] = (
            lorth_reference_sigma
            * (lorth_reference / lorth_values[positive_lorth]) ** 2
        )
    else:
        lorth_reference_index = 0

    dimension_reference_index = int(np.argmin(normal_dimensions))
    dimension_reference_normal = float(
        normal_dimensions[dimension_reference_index]
    )
    dimension_reference_sigma = float(
        experiments["latent_dim"]["summary"]["mean_optimum"]
        [dimension_reference_index]
    )
    dimension_dominant_prediction = (
        dimension_reference_sigma
        * dimension_reference_normal
        / normal_dimensions
    )

    fit_diagnostics = {
        "rho_full_quadratic": regression_diagnostics(
            experiments["rho"]["summary"]["mean_optimum"],
            rho_prediction,
        ),
        "lorth_reciprocal": regression_diagnostics(
            experiments["lorth"]["summary"]["mean_optimum"],
            lorth_prediction,
        ),
        "dimension_reciprocal": regression_diagnostics(
            experiments["latent_dim"]["summary"]["mean_optimum"],
            dimension_prediction,
        ),
    }

    stability_rows: list[dict[str, object]] = []
    for key, experiment in experiments.items():
        stability_rows.extend(
            grid_stability_rows(
                spec.key,
                key,
                experiment["values"],
                sigma_grid,
                experiment["curves"],
            )
        )
    stability_rows.extend(
        grid_stability_rows(
            spec.key,
            "ambient_smoothing",
            np.asarray([args.ambient_dimension], dtype=float),
            sigma_grid,
            ambient_curves[:, None, :],
        )
    )

    return {
        "spec": spec,
        "ambient_dimension": args.ambient_dimension,
        "sigma_grid": sigma_grid,
        "display_sigma_max": args.display_sigma_max,
        "confidence_level": args.confidence_level,
        "reference_dimension": reference_dimension,
        "experiments": experiments,
        "ambient_smoothing": {
            "curves": ambient_curves,
            "summary": ambient_summary,
        },
        "verification_rows": verification_rows,
        "rho_prediction": rho_prediction,
        "rho_prediction_normalized": rho_prediction_normalized,
        "rho_beta": rho_beta,
        "rho_beta_fitted": rho_beta_fitted,
        "rho_normalized": rho_normalized,
        "rho_empirical_slope": rho_empirical_slope,
        "rho_empirical_slope_ci": rho_slope_ci,
        "rho_raw_slopes": rho_raw_slopes,
        "rho_slope_std": rho_slope_std,
        "rho_theoretical_slope": 1.0,
        "w0_estimates": w0_estimates,
        "w0_mean": w0_mean,
        "w0_std": w0_std,
        "lorth_prediction": lorth_prediction,
        "lorth_dominant_prediction": lorth_dominant_prediction,
        "lorth_dominant_reference_index": lorth_reference_index,
        "lorth_q": lorth_q,
        "dimension_prediction": dimension_prediction,
        "dimension_dominant_prediction": dimension_dominant_prediction,
        "dimension_q": dimension_q,
        "fit_diagnostics": fit_diagnostics,
        "grid_stability_rows": stability_rows,
        "reference_normal_dimension": args.reference_normal_dimension,
        "slope_study_lorth": args.slope_study_lorth,
        "dimension_lorth": args.dimension_lorth,
    }


# =============================================================================
# 6. Figures
# =============================================================================
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


def save_figure(figure: plt.Figure, base_path: Path) -> None:
    for suffix in (".pdf", ".png"):
        output = base_path.with_suffix(suffix)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.unlink(missing_ok=True)
        figure.savefig(
            temporary,
            format=suffix.removeprefix("."),
            bbox_inches="tight",
        )
        if suffix == ".pdf":
            with temporary.open("rb") as handle:
                handle.seek(max(0, temporary.stat().st_size - 1024))
                if b"%%EOF" not in handle.read():
                    temporary.unlink(missing_ok=True)
                    raise OSError(f"Incomplete PDF figure: {output}")
        temporary.replace(output)
    plt.close(figure)


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


def centered_curve_summary(
    curves: Array,
    confidence_level: float,
) -> tuple[Array, Array]:
    """Mean and confidence half-width after run-wise centering at sigma=0."""
    centered = curves - curves[..., [0]]
    mean = np.mean(centered, axis=0)
    ci_halfwidth = t_confidence_halfwidth(
        centered,
        confidence_level,
        axis=0,
    )
    return mean, ci_halfwidth


def local_sigma_limit(
    sigma: Array,
    mean_optima: Array,
    display_sigma_max: float,
) -> float:
    """Choose an automatic window for displaying the initial variation.

    Minima near the edge of the displayed interval are excluded when a
    genuinely local group of minima exists.  This prevents one broad curve
    from destroying the resolution of the small-bandwidth inset.
    """
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
    result: dict[str, object],
    key: str,
    directory: Path,
    include_ambient: bool,
) -> None:
    experiment = result["experiments"][key]
    values = np.asarray(experiment["values"])
    summary = experiment["summary"]
    sigma = result["sigma_grid"]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(values)))
    centered_mean, centered_ci = centered_curve_summary(
        experiment["curves"],
        result["confidence_level"],
    )
    for index, (value, color) in enumerate(zip(values, colors)):
        mean = summary["mean_curve"][index]
        ci = summary["curve_ci_halfwidth"][index]
        axis.plot(sigma, mean, color=color, label=parameter_label(key, value))
        axis.fill_between(
            sigma,
            mean - ci,
            mean + ci,
            color=color,
            alpha=0.08,
            linewidth=0.0,
        )
        optimum = float(summary["mean_optimum"][index])
        axis.plot(
            optimum,
            np.interp(optimum, sigma, mean),
            marker="o",
            markersize=5.5,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )

    ambient_summary = result["ambient_smoothing"]["summary"]
    ambient_mean = ambient_summary["mean_curve"][0]
    ambient_ci = ambient_summary["curve_ci_halfwidth"][0]
    if include_ambient:
        # Draw the reference after the latent curves so the dashed line is
        # visible wherever the curves overlap.
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
            label=rf"Ambient",
            zorder=3,
        )
        ambient_optimum = float(ambient_summary["mean_optimum"][0])
        axis.plot(
            ambient_optimum,
            np.interp(ambient_optimum, sigma, ambient_mean),
            marker="o",
            markersize=5.5,
            markerfacecolor="black",
            markeredgecolor="white",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=4,
        )

    # The inset displays the change from the unsmoothed value.  The centering
    # is performed separately within each Monte Carlo repetition before the
    # mean and standard deviation are taken.
    #inset = axis.inset_axes([0.49, 0.12, 0.48, 0.38])
    inset = axis.inset_axes([0.15, 0.55, 0.46, 0.38]) 

    inset.set_zorder(20)
    zoom_sigma = local_sigma_limit(
        sigma,
        summary["mean_optimum"],
        result["display_sigma_max"],
    )
    zoom_mask = sigma <= zoom_sigma
    centered_ambient_mean, centered_ambient_ci = centered_curve_summary(
        result["ambient_smoothing"]["curves"],
        result["confidence_level"],
    )
    for index, color in enumerate(colors):
        inset.plot(
            sigma[zoom_mask],
            centered_mean[index, zoom_mask],
            color=color,
            linewidth=1.6,
        )
        inset.fill_between(
            sigma[zoom_mask],
            centered_mean[index, zoom_mask]
            - centered_ci[index, zoom_mask],
            centered_mean[index, zoom_mask]
            + centered_ci[index, zoom_mask],
            color=color,
            alpha=0.06,
            linewidth=0.0,
        )
        optimum = float(summary["mean_optimum"][index])
        if optimum <= zoom_sigma:
            inset.plot(
                optimum,
                np.interp(optimum, sigma, centered_mean[index]),
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
        if ambient_optimum <= zoom_sigma:
            inset.plot(
                ambient_optimum,
                np.interp(ambient_optimum, sigma, centered_ambient_mean),
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

    axis.set_xlabel(r"Bandwidth $\sigma$")
    axis.set_ylabel(r"$W_2^2$")
    axis.set_xlim(0.0, result["display_sigma_max"])
    axis.grid(alpha=0.25)
    display_mask = sigma <= result["display_sigma_max"]

    y_min = np.min(
        (
            summary["mean_curve"] - summary["curve_ci_halfwidth"]
        )[:, display_mask]
    )
    y_max = np.max(
        (
            summary["mean_curve"] + summary["curve_ci_halfwidth"]
        )[:, display_mask]
    )

    if include_ambient:
        y_min = min(
            y_min,
            np.min(
                (
                    ambient_summary["mean_curve"][0]
                    - ambient_summary["curve_ci_halfwidth"][0]
                )[display_mask]
            ),
        )
        y_max = max(
            y_max,
            np.max(
                (
                    ambient_summary["mean_curve"][0]
                    + ambient_summary["curve_ci_halfwidth"][0]
                )[display_mask]
            ),
        )

    # Add 5% padding to bounds (anchors lower bound at 0.0 if y_min is positive)
    y_margin = 0.05 * (y_max - y_min)
    axis.set_ylim(min(0.0, y_min - y_margin), y_max + y_margin)
    #axis.legend(frameon=False, ncol=2)
    #axis.legend(frameon=False, ncol=1, loc="upper right")
    axis.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02,0.5), ncol=1)

    figure.tight_layout(rect=[0,0,0.78,1])
    suffix = "with_ambient" if include_ambient else "latent_only"
    save_figure(
        figure,
        directory
        / f"{result['spec'].key}_explicit_w2_vs_sigma_{key}_{suffix}",
    )


def plot_optimal_sigma(
    result: dict[str, object], key: str, directory: Path
) -> None:
    experiment = result["experiments"][key]
    values = np.asarray(experiment["values"], dtype=float)
    summary = experiment["summary"]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))

    if key == "rho":
        observed_mean = result["rho_normalized"]["mean"]
        observed_error = result["rho_normalized"]["ci_halfwidth"]
    else:
        observed_mean = summary["mean_optimum"]
        observed_error = summary["optimum_ci_halfwidth"]

    axis.errorbar(
        values,
        observed_mean,
        yerr=observed_error,
        color="#1f6fba",
        marker="o",
        capsize=4,
        label="Observed",
        zorder=3,
    )

    if key == "rho":
        dense = np.linspace(values[0], values[-1], 200)
        empirical_slope = result["rho_empirical_slope"]
        slope_ci = result["rho_empirical_slope_ci"]
        axis.plot(
            dense,
            1.0 + empirical_slope * (dense - 1.0),
            color="#6a51a3",
            linestyle="--",
            linewidth=2.8,
            label=(
                rf"Linear fit (slope={empirical_slope:.3f}"
                rf"$\pm${slope_ci:.3f})"
            ),
            zorder=5,
        )
        beta = float(result["rho_beta"])
        dense_full = dense * (1.0 + beta) / (dense**2 + beta)
        beta_source = "fitted" if result["rho_beta_fitted"] else "fixed"
        axis.plot(
            dense,
            dense_full,
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=rf"Full quadratic ({beta_source} $\beta={beta:.3g}$)",
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
        reference_index = int(np.argmin(np.abs(values)))
        reference_sigma = float(summary["mean_optimum"][reference_index])
        q_value = float(result["lorth_q"])
        dense_prediction = (
            reference_sigma
            * q_value
            / (result["reference_normal_dimension"] * dense**2 + q_value)
        )
        axis.plot(
            dense,
            dense_prediction,
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=(
                r"Descriptive fit: $a/(L_{\rm orth}^2(d-m)+q)$"
                + rf", $q={q_value:.3g}$"
            ),
            zorder=6,
        )
        dominant = np.asarray(result["lorth_dominant_prediction"])
        valid = np.isfinite(dominant)
        axis.plot(
            values[valid],
            dominant[valid],
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
        intrinsic_dimension = result["spec"].intrinsic_dimension
        dense_penalty = (
            result["dimension_lorth"] ** 2
            * (dense - intrinsic_dimension)
        )
        observed_penalties = (
            result["dimension_lorth"] ** 2
            * (values - intrinsic_dimension)
        )
        reference_index = int(np.argmin(observed_penalties))
        reference_sigma = float(summary["mean_optimum"][reference_index])
        reference_penalty = float(observed_penalties[reference_index])
        q_value = float(result["dimension_q"])
        dense_prediction = (
            reference_sigma
            * (reference_penalty + q_value)
            / (dense_penalty + q_value)
        )
        axis.plot(
            dense,
            dense_prediction,
            color="0.35",
            linestyle="-.",
            linewidth=2.4,
            label=(
                r"Descriptive fit: $a/(L_{\rm orth}^2(d-m)+q)$"
                + rf", $q={q_value:.3g}$"
            ),
            zorder=6,
        )
        axis.plot(
            values,
            result["dimension_dominant_prediction"],
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
    legend_order = sorted(
        range(len(labels)),
        key=lambda index: 0 if labels[index] == "Observed" else 1,
    )
    axis.legend(
        [handles[index] for index in legend_order],
        [labels[index] for index in legend_order],
        frameon=False,
    )
    figure.tight_layout()
    save_figure(
        figure,
        directory / f"{result['spec'].key}_explicit_optimal_sigma_vs_{key}",
    )


def plot_parameter_verification(
    result: dict[str, object], directory: Path
) -> None:
    rows = result["verification_rows"]
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 8.0))
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
                    row["latent_dimension"]
                    - row["intrinsic_dimension"]
                    for row in selected
                ],
                dtype=float,
            )
        else:
            raw_x = np.asarray(
                [row["prescribed_value"] for row in selected],
                dtype=float,
            )
        raw_y = np.asarray([row["measured_value"] for row in selected])
        x = np.unique(raw_x)
        y = np.asarray(
            [np.mean(raw_y[np.isclose(raw_x, value)]) for value in x]
        )
        yerr = np.asarray(
            [
                t_confidence_halfwidth(
                    raw_y[np.isclose(raw_x, value)],
                    result["confidence_level"],
                    axis=0,
                )
                for value in x
            ],
            dtype=float,
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
    save_figure(
        figure,
        directory / f"{result['spec'].key}_explicit_parameter_verification",
    )


def plot_grid_stability(
    result: dict[str, object],
    directory: Path,
) -> None:
    """Appendix diagnostic for bandwidth-grid sensitivity."""
    rows = [
        row
        for row in result["grid_stability_rows"]
        if row["coarse_stride"] == 4
    ]
    figure, axis = plt.subplots(figsize=(7.2, 5.0))
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
    for key in colors:
        selected = [row for row in rows if row["parameter"] == key]
        if not selected:
            continue
        axis.scatter(
            [row["full_optimal_sigma"] for row in selected],
            [row["coarse_optimal_sigma"] for row in selected],
            color=colors[key],
            alpha=0.75,
            label=labels[key],
            zorder=3,
        )
    maximum = max(
        max(row["full_optimal_sigma"], row["coarse_optimal_sigma"])
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
    save_figure(
        figure,
        directory / f"{result['spec'].key}_bandwidth_grid_stability",
    )


def create_figures(results: list[dict[str, object]], output_root: Path) -> None:
    configure_plot_style()
    for result in results:
        directory = output_root / result["spec"].key
        directory.mkdir(parents=True, exist_ok=True)
        for key in ("rho", "delta", "lorth", "latent_dim"):
            plot_w2_curves(result, key, directory, include_ambient=False)
            plot_w2_curves(result, key, directory, include_ambient=True)
            if key != "delta":
                plot_optimal_sigma(result, key, directory)
        plot_parameter_verification(result, directory)
        plot_grid_stability(result, directory)


# =============================================================================
# 7. CSV output and terminal summary
# =============================================================================
def save_csv_outputs(results: list[dict[str, object]], output_root: Path) -> None:
    verification_rows = [
        row for result in results for row in result["verification_rows"]
    ]
    with (output_root / "explicit_parameter_verification.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(verification_rows[0]),
        )
        writer.writeheader()
        writer.writerows(verification_rows)

    with (output_root / "explicit_w2_curves.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "parameter",
                "parameter_value",
                "sigma",
                "w2_squared_mean",
                "w2_squared_std",
                "w2_squared_ci_halfwidth",
            ]
        )
        for result in results:
            for key, experiment in result["experiments"].items():
                for parameter_index, value in enumerate(experiment["values"]):
                    for sigma_index, sigma in enumerate(result["sigma_grid"]):
                        writer.writerow(
                            [
                                result["spec"].key,
                                key,
                                float(value),
                                float(sigma),
                                float(
                                    experiment["summary"]["mean_curve"][
                                        parameter_index, sigma_index
                                    ]
                                ),
                                float(
                                    experiment["summary"]["std_curve"][
                                        parameter_index, sigma_index
                                    ]
                                ),
                                float(
                                    experiment["summary"]
                                    ["curve_ci_halfwidth"]
                                    [parameter_index, sigma_index]
                                ),
                            ]
                        )
            ambient_summary = result["ambient_smoothing"]["summary"]
            for sigma_index, sigma in enumerate(result["sigma_grid"]):
                writer.writerow(
                    [
                        result["spec"].key,
                        "ambient_smoothing",
                        int(result["ambient_dimension"]),
                        float(sigma),
                        float(ambient_summary["mean_curve"][0, sigma_index]),
                        float(ambient_summary["std_curve"][0, sigma_index]),
                        float(
                            ambient_summary["curve_ci_halfwidth"]
                            [0, sigma_index]
                        ),
                    ]
                )

    with (output_root / "explicit_w2_curves_by_run.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "repetition",
                "parameter",
                "parameter_value",
                "sigma",
                "w2_squared",
            ]
        )
        for result in results:
            for key, experiment in result["experiments"].items():
                for repetition in range(experiment["curves"].shape[0]):
                    for parameter_index, value in enumerate(
                        experiment["values"]
                    ):
                        for sigma_index, sigma in enumerate(
                            result["sigma_grid"]
                        ):
                            writer.writerow(
                                [
                                    result["spec"].key,
                                    repetition,
                                    key,
                                    float(value),
                                    float(sigma),
                                    float(
                                        experiment["curves"]
                                        [repetition, parameter_index, sigma_index]
                                    ),
                                ]
                            )
            for repetition in range(
                result["ambient_smoothing"]["curves"].shape[0]
            ):
                for sigma_index, sigma in enumerate(result["sigma_grid"]):
                    writer.writerow(
                        [
                            result["spec"].key,
                            repetition,
                            "ambient_smoothing",
                            float(result["ambient_dimension"]),
                            float(sigma),
                            float(
                                result["ambient_smoothing"]["curves"]
                                [repetition, sigma_index]
                            ),
                        ]
                    )

    with (output_root / "explicit_optimal_sigma.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "parameter",
                "parameter_value",
                "optimal_sigma_mean",
                "optimal_sigma_std",
                "optimal_sigma_ci_halfwidth",
                "descriptive_prediction",
                "dominant_term_prediction",
                "minimum_w2_squared_mean",
                "minimum_w2_squared_std",
                "minimum_w2_squared_ci_halfwidth",
                "w2_squared_at_sigma_zero",
                "absolute_improvement_at_optimum",
                "improvement_ci_halfwidth",
                "relative_improvement_percent",
                "rho_beta_B_over_A",
                "rho_beta_source",
            ]
        )
        for result in results:
            for key, experiment in result["experiments"].items():
                if key == "rho":
                    prediction = result["rho_prediction"]
                    dominant_prediction = (
                        experiment["summary"]["mean_optimum"]
                        [int(np.argmin(np.abs(experiment["values"] - 1.0)))]
                        * np.asarray(experiment["values"])
                    )
                elif key == "delta":
                    prediction = None
                    dominant_prediction = None
                elif key == "lorth":
                    prediction = result["lorth_prediction"]
                    dominant_prediction = result["lorth_dominant_prediction"]
                elif key == "latent_dim":
                    prediction = result["dimension_prediction"]
                    dominant_prediction = result[
                        "dimension_dominant_prediction"
                    ]
                else:
                    prediction = None
                    dominant_prediction = None
                for index, value in enumerate(experiment["values"]):
                    summary = experiment["summary"]
                    baseline = float(summary["mean_curve"][index, 0])
                    minimum = float(summary["mean_minimum"][index])
                    run_improvements = (
                        experiment["curves"][:, index, 0]
                        - summary["minima"][:, index]
                    )
                    improvement = float(np.mean(run_improvements))
                    improvement_ci = float(
                        t_confidence_halfwidth(
                            run_improvements,
                            result["confidence_level"],
                            axis=0,
                        )
                    )
                    writer.writerow(
                        [
                            result["spec"].key,
                            key,
                            float(value),
                            float(summary["mean_optimum"][index]),
                            float(summary["std_optimum"][index]),
                            float(summary["optimum_ci_halfwidth"][index]),
                            (
                                float(prediction[index])
                                if prediction is not None
                                else ""
                            ),
                            (
                                float(dominant_prediction[index])
                                if dominant_prediction is not None
                                and np.isfinite(dominant_prediction[index])
                                else ""
                            ),
                            minimum,
                            float(summary["std_minimum"][index]),
                            float(summary["minimum_ci_halfwidth"][index]),
                            baseline,
                            improvement,
                            improvement_ci,
                            (
                                100.0 * improvement / baseline
                                if baseline > 0.0
                                else ""
                            ),
                            (
                                float(result["rho_beta"])
                                if key == "rho"
                                else ""
                            ),
                            (
                                "fitted"
                                if key == "rho"
                                and result["rho_beta_fitted"]
                                else (
                                    "supplied"
                                    if key == "rho"
                                    else ""
                                )
                            ),
                        ]
                    )

    with (output_root / "explicit_optimal_sigma_by_run.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "repetition",
                "parameter",
                "parameter_value",
                "optimal_sigma",
                "minimum_w2_squared",
                "boundary_minimum",
            ]
        )
        for result in results:
            for key, experiment in result["experiments"].items():
                for repetition in range(experiment["curves"].shape[0]):
                    for index, value in enumerate(experiment["values"]):
                        grid_index = int(
                            np.argmin(experiment["curves"][repetition, index])
                        )
                        writer.writerow(
                            [
                                result["spec"].key,
                                repetition,
                                key,
                                float(value),
                                float(experiment["summary"]["optima"]
                                      [repetition, index]),
                                float(experiment["summary"]["minima"]
                                      [repetition, index]),
                                grid_index in (0, len(result["sigma_grid"]) - 1),
                            ]
                        )

    with (output_root / "explicit_normalized_optima.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "parameter",
                "parameter_value",
                "normalized_optimal_sigma_mean",
                "normalized_optimal_sigma_std",
                "normalized_optimal_sigma_ci_halfwidth",
                "normalization_method",
                "reference_optimal_sigma_mean",
                "zero_reference_run_count",
            ]
        )
        for result in results:
            experiment = result["experiments"]["rho"]
            normalized = result["rho_normalized"]
            for index, value in enumerate(experiment["values"]):
                writer.writerow(
                    [
                        result["spec"].key,
                        "rho",
                        float(value),
                        float(normalized["mean"][index]),
                        float(normalized["std"][index]),
                        float(normalized["ci_halfwidth"][index]),
                        normalized["method"],
                        float(normalized["reference_mean"]),
                        int(normalized["zero_reference_run_count"]),
                    ]
                )

    stability_rows = [
        row for result in results for row in result["grid_stability_rows"]
    ]
    with (output_root / "explicit_bandwidth_grid_stability.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(stability_rows[0]))
        writer.writeheader()
        writer.writerows(stability_rows)

    with (output_root / "explicit_fit_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "fit",
                "rmse",
                "r_squared",
                "max_absolute_error",
                "nuisance_parameter",
                "nuisance_value",
            ]
        )
        for result in results:
            for fit_name, diagnostics in result["fit_diagnostics"].items():
                if fit_name == "rho_full_quadratic":
                    parameter_name = "beta"
                    parameter_value = result["rho_beta"]
                elif fit_name == "lorth_reciprocal":
                    parameter_name = "q"
                    parameter_value = result["lorth_q"]
                else:
                    parameter_name = "q"
                    parameter_value = result["dimension_q"]
                writer.writerow(
                    [
                        result["spec"].key,
                        fit_name,
                        diagnostics["rmse"],
                        diagnostics["r_squared"],
                        diagnostics["max_absolute_error"],
                        parameter_name,
                        parameter_value,
                    ]
                )

    with (output_root / "explicit_rho_slope_by_run.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "manifold",
                "repetition",
                "raw_slope",
                "reference_optimal_sigma",
            ]
        )
        for result in results:
            reference_index = int(
                np.argmin(
                    np.abs(result["experiments"]["rho"]["values"] - 1.0)
                )
            )
            reference_optima = result["experiments"]["rho"]["summary"][
                "optima"
            ][:, reference_index]
            for repetition, (slope, reference) in enumerate(
                zip(result["rho_raw_slopes"], reference_optima)
            ):
                writer.writerow(
                    [
                        result["spec"].key,
                        repetition,
                        float(slope),
                        float(reference),
                    ]
                )


def print_summary(results: list[dict[str, object]], args: argparse.Namespace) -> None:
    print("\nExplicit fixed-D encoder-decoder experiment")
    print(f"  ambient dimension D       : {args.ambient_dimension}")
    print(f"  empirical anchors n       : {args.n_empirical}")
    print(f"  OT samples                : {args.n_ot}")
    print(f"  repetitions               : {args.repetitions}")
    print(
        f"  uncertainty               : "
        f"{100.0 * args.confidence_level:.1f}% t confidence intervals"
    )
    print(
        f"  sigma grid                : [0, {args.sigma_max:g}], "
        f"{args.sigma_points} points"
    )
    for result in results:
        rows = result["verification_rows"]
        max_reconstruction = max(
            row["reconstruction_parameter_error"] for row in rows
        )
        max_tangent = max(row["tangent_map_max_error"] for row in rows)
        max_orthogonal = max(
            row["orthogonal_map_max_error"] for row in rows
        )
        max_encoder = max(row["encoder_gram_max_error"] for row in rows)
        print(f"\n  {result['spec'].latex_name}")
        print(
            "    maximum reconstruction identity error : "
            f"{max_reconstruction:.3e}"
        )
        print(
            "    maximum tangential Jacobian error      : "
            f"{max_tangent:.3e}"
        )
        print(
            "    maximum orthogonal Jacobian error      : "
            f"{max_orthogonal:.3e}"
        )
        print(
            "    maximum encoder Gram error             : "
            f"{max_encoder:.3e}"
        )
        beta_source = (
            "fitted" if result["rho_beta_fitted"] else "supplied"
        )
        print(
            "    rho quadratic beta = B_n/A_n           : "
            f"{result['rho_beta']:.6f} ({beta_source})"
        )
        dominant_penalty = (
            result["slope_study_lorth"] ** 2
            * result["reference_normal_dimension"]
        )
        print(
            "    intrinsic W_0 estimate, mean +/- SD    : "
            f"{result['w0_mean']:.6f} +/- {result['w0_std']:.6f}"
        )
        print(
            "    fixed slope-study penalty              : "
            f"L_orth^2(d-m)={dominant_penalty:.6f}"
        )
        print(
            "    rho normalized slope, observed/theory  : "
            f"{result['rho_empirical_slope']:.6f} +/- "
            f"{result['rho_empirical_slope_ci']:.6f} / "
            f"{result['rho_theoretical_slope']:.6f}"
        )
        print(
            "    rho=1 runs with boundary optimum zero  : "
            f"{result['rho_normalized']['zero_reference_run_count']} / "
            f"{args.repetitions}"
        )
        print(
            "    reciprocal-fit q, L_orth / dimension   : "
            f"{result['lorth_q']:.6f} / {result['dimension_q']:.6f}"
        )
        for fit_name, diagnostics in result["fit_diagnostics"].items():
            print(
                f"    fit {fit_name:<22}: "
                f"R^2={diagnostics['r_squared']:.4f}, "
                f"RMSE={diagnostics['rmse']:.3e}"
            )
        stability = result["grid_stability_rows"]
        finite_relative = [
            row["relative_change"]
            for row in stability
            if np.isfinite(row["relative_change"])
        ]
        boundary_count = sum(
            bool(row["full_grid_boundary_minimum"])
            for row in stability
            if row["coarse_stride"] == 2
        )
        print(
            "    maximum relative grid-change          : "
            f"{max(finite_relative, default=float('nan')):.3e}"
        )
        print(
            "    full-grid boundary minima             : "
            f"{boundary_count}"
        )
        for key in ("rho", "delta", "lorth", "latent_dim"):
            experiment = result["experiments"][key]
            print(f"    {key}:")
            for index, value in enumerate(experiment["values"]):
                summary = experiment["summary"]
                run_improvements = (
                    experiment["curves"][:, index, 0]
                    - summary["minima"][:, index]
                )
                improvement_mean = float(np.mean(run_improvements))
                improvement_ci = float(
                    t_confidence_halfwidth(
                        run_improvements,
                        args.confidence_level,
                        axis=0,
                    )
                )
                print(
                    f"      {float(value):>7g}: "
                    f"sigma*={summary['mean_optimum'][index]:.6f} "
                    f"+/- {summary['optimum_ci_halfwidth'][index]:.6f}; "
                    f"min W2^2={summary['mean_minimum'][index]:.6f}; "
                    f"improvement={improvement_mean:.6f} "
                    f"+/- {improvement_ci:.6f}"
                )


def save_run_metadata(
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    arguments = {
        key: (str(value) if isinstance(value, Path) else value)
        for key, value in vars(args).items()
    }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": matplotlib.__version__,
            "POT": getattr(ot, "__version__", "unknown"),
        },
        "arguments": arguments,
    }
    with (output_root / "run_metadata.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


# =============================================================================
# 8. Command line
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifolds",
        nargs="+",
        choices=[spec.key for spec in MANIFOLDS],
        default=[spec.key for spec in MANIFOLDS],
        help=(
            "Manifolds to run. Use `--manifolds sphere` or "
            "`--manifolds product` to split a long computation."
        ),
    )
    parser.add_argument("--ambient-dimension", type=int, default=100)
    parser.add_argument("--n-empirical", type=int, default=500)
    parser.add_argument("--n-ot", type=int, default=200)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Two-sided Student-t confidence level for plotted uncertainty.",
    )
    parser.add_argument("--sigma-max", type=float, default=0.1)
    parser.add_argument("--sigma-points", type=int, default=81)
    parser.add_argument(
        "--sigma-grid-power",
        type=float,
        default=2.0,
        help=(
            "Power for the grid sigma_j=sigma_max*(j/(N-1))^power. "
            "A value larger than one resolves small optimal bandwidths."
        ),
    )
    parser.add_argument("--display-sigma-max", type=float, default=0.1)
    parser.add_argument(
        "--rho-values",
        type=float,
        nargs="+",
        default=[0.80, 0.90, 1.00, 1.10, 1.20],
    )
    parser.add_argument(
        "--rho-beta",
        type=float,
        default=None,
        help=(
            "Optional prescribed beta=B_n/A_n in the full rho prediction. "
            "If omitted, beta is fitted nonnegatively from the observed "
            "optimal bandwidths."
        ),
    )
    parser.add_argument(
        "--delta-values",
        type=float,
        nargs="+",
        default=[0.00, 0.2, 0.4, 0.6, 0.8],
    )
    parser.add_argument(
        "--lorth-values",
        type=float,
        nargs="+",
        default=[0.00, 0.15, 0.25, 0.35, 0.50],
    )
    parser.add_argument(
        "--normal-dimensions",
        type=int,
        nargs="+",
        default=[10, 25, 40, 60, 80],
        help="Values of d-m in the latent-dimension experiment.",
    )
    parser.add_argument(
        "--reference-normal-dimension",
        type=int,
        default=80,
        help="Fixed d-m in the rho, delta, and L_orth experiments.",
    )
    parser.add_argument(
        "--slope-study-lorth",
        type=float,
        default=1.00,
        help=(
            "Fixed L_orth in the rho and delta slope experiments. "
            "Together with reference-normal-dimension, this makes "
            "L_orth^2(d-m) dominate the quadratic coefficient."
        ),
    )
    parser.add_argument("--dimension-lorth", type=float, default=0.50)
    parser.add_argument("--shepard-power", type=int, default=2)
    parser.add_argument("--finite-difference-step", type=float, default=2.0e-5)
    parser.add_argument("--checked-anchors", type=int, default=10)
    parser.add_argument("--decode-chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("explicit_global_encoder_decoder_results"),
    )
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.quick:
        args.n_empirical = min(args.n_empirical, 20)
        args.n_ot = min(args.n_ot, 40)
        args.repetitions = 1
        args.sigma_points = min(args.sigma_points, 15)
        args.checked_anchors = min(args.checked_anchors, 3)
        args.rho_values = [0.8, 1.0, 1.2]
        args.delta_values = [0.0, 0.4, 0.8]
        args.lorth_values = [0.0, 0.25, 0.5]
        args.normal_dimensions = [10, 40, 80]
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.ambient_dimension < 5:
        raise ValueError("ambient-dimension must be at least 5.")
    if args.n_empirical < 2 or args.n_ot < 2:
        raise ValueError("n-empirical and n-ot must both be at least two.")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if not 0.0 < args.confidence_level < 1.0:
        raise ValueError("confidence-level must lie strictly between 0 and 1.")
    if args.sigma_max <= 0.0 or args.sigma_points < 5:
        raise ValueError("The sigma grid is invalid.")
    if args.sigma_grid_power < 1.0:
        raise ValueError("sigma-grid-power must be at least one.")
    if not 0.0 < args.display_sigma_max <= args.sigma_max:
        raise ValueError("display-sigma-max must lie in (0, sigma-max].")
    if np.any(np.asarray(args.rho_values) <= 0.0):
        raise ValueError("rho-values must be positive.")
    if not np.any(np.isclose(args.rho_values, 1.0)):
        raise ValueError("rho-values must include 1.")
    if args.rho_beta is not None and args.rho_beta < 0.0:
        raise ValueError("rho-beta must be nonnegative.")
    if np.any(np.asarray(args.delta_values) < 0.0):
        raise ValueError("delta-values must be nonnegative.")
    if not np.any(np.isclose(args.delta_values, 0.0)):
        raise ValueError("delta-values must include 0.")
    if np.any(np.asarray(args.lorth_values) < 0.0):
        raise ValueError("lorth-values must be nonnegative.")
    if not np.any(np.isclose(args.lorth_values, 0.0)):
        raise ValueError("lorth-values must include 0.")
    if np.any(np.asarray(args.normal_dimensions) < 2):
        raise ValueError("normal-dimensions must be at least 2.")
    if max(args.normal_dimensions) + 3 > args.ambient_dimension:
        raise ValueError("Some requested product latent dimensions exceed D.")
    if args.reference_normal_dimension + 3 > args.ambient_dimension:
        raise ValueError("The reference product latent dimension exceeds D.")
    if args.dimension_lorth < 0.0:
        raise ValueError("dimension-lorth must be nonnegative.")
    if args.slope_study_lorth < 0.0:
        raise ValueError("slope-study-lorth must be nonnegative.")
    if args.shepard_power < 1:
        raise ValueError("shepard-power must be a positive integer.")
    if args.finite_difference_step <= 0.0:
        raise ValueError("finite-difference-step must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = [spec for spec in MANIFOLDS if spec.key in args.manifolds]
    results = [run_manifold_experiment(spec, args) for spec in selected]
    create_figures(results, args.output_dir)
    save_csv_outputs(results, args.output_dir)
    save_run_metadata(args, args.output_dir)
    print_summary(results, args)
    print(f"\nSaved results to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
