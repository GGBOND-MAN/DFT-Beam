from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import math
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import erf, i0e


np.set_printoptions(precision=5, suppress=True)

C0 = 3.0e8


@dataclass(frozen=True)
class SimulationConfig:
    N: int = 128
    fc: float = 100e9
    r_min: float = 0.8
    r_max: float = 12.0
    d: float | None = None
    amplitude_factor: float = 1.0
    dft_oversampling: int = 1

    @property
    def wavelength(self) -> float:
        return C0 / self.fc

    @property
    def spacing(self) -> float:
        return self.wavelength / 2 if self.d is None else self.d

    @property
    def aperture(self) -> float:
        return (self.N - 1) * self.spacing

    @property
    def fresnel_distance(self) -> float:
        return 0.5 * np.sqrt(self.aperture**3 / self.wavelength)

    @property
    def rayleigh_distance(self) -> float:
        return 2.0 * self.aperture**2 / self.wavelength

    def modified_rayleigh(self, u: float, p: float = 3.0) -> float:
        # Paper Eq. (23), rho=0.5.
        return self.N**2 * self.spacing * (1.0 - u**2) / (2.0 * p)


@dataclass
class Estimate:
    method: str
    u_hat: float
    r_hat: float
    overhead: int
    details: dict[str, Any] = field(default_factory=dict)


QUICK_CFG = SimulationConfig(N=128, r_min=0.8, r_max=12.0)
PAPER_CFG = SimulationConfig(N=512, r_min=2.0, r_max=100.0)


def element_offsets(cfg: SimulationConfig) -> np.ndarray:
    return (np.arange(cfg.N) - (cfg.N - 1) / 2.0) * cfg.spacing


def near_field_steering(cfg: SimulationConfig, u: float, r: float) -> np.ndarray:
    '''Exact normalized spherical-wave ULA steering vector.'''
    u = float(np.clip(u, -0.999999, 0.999999))
    r = float(max(r, 1e-9))
    y = element_offsets(cfg)
    rn = np.sqrt(r**2 + y**2 - 2.0 * r * u * y)
    return np.exp(-1j * 2.0 * np.pi * (rn - r) / cfg.wavelength) / np.sqrt(cfg.N)


def far_field_steering(cfg: SimulationConfig, u: float) -> np.ndarray:
    u = float(np.clip(u, -1.0, 1.0))
    n = np.arange(cfg.N) - (cfg.N - 1) / 2.0
    return np.exp(1j * 2.0 * np.pi * cfg.spacing / cfg.wavelength * n * u) / np.sqrt(cfg.N)


def dft_codebook(cfg: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    K = cfg.N * cfg.dft_oversampling
    u_grid = -1.0 + (2.0 * np.arange(K) + 1.0) / K
    W = np.vstack([far_field_steering(cfg, u) for u in u_grid])
    return u_grid, W


def complex_measurements(
    h: np.ndarray,
    beams: np.ndarray,
    sigma2: float,
    rng: np.random.Generator,
    path_gain: float,
) -> np.ndarray:
    beams = np.atleast_2d(beams)
    clean = path_gain * (beams @ np.conjugate(h))
    noise = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal(clean.size) + 1j * rng.standard_normal(clean.size)
    )
    return clean + noise


def amplitude_measurements(*args, **kwargs) -> np.ndarray:
    return np.abs(complex_measurements(*args, **kwargs))


def reference_noise_variance(cfg: SimulationConfig, snr_db: float, reference_r: float = 5.0) -> float:
    reference_power = (cfg.amplitude_factor / reference_r) ** 2
    return reference_power / (10.0 ** (snr_db / 10.0))


def path_gain(cfg: SimulationConfig, r: float) -> float:
    # Paper Eq. (1): h^H = sqrt(N) * g * b^H.  amplitude_factor represents
    # the distance-independent part of g; the steering vector b is unit norm.
    return np.sqrt(cfg.N) * cfg.amplitude_factor / float(r)


def predicted_dft_amplitudes(cfg: SimulationConfig, u: float, r: float, W_dft: np.ndarray) -> np.ndarray:
    h = near_field_steering(cfg, u, r)
    return np.abs(W_dft @ np.conjugate(h))


def achievable_rate(
    cfg: SimulationConfig,
    true_u: float,
    true_r: float,
    estimate: Estimate,
    sigma2: float,
) -> float:
    h = near_field_steering(cfg, true_u, true_r)
    w = near_field_steering(cfg, estimate.u_hat, estimate.r_hat)
    signal = path_gain(cfg, true_r) ** 2 * np.abs(np.vdot(h, w)) ** 2
    return float(np.log2(1.0 + signal / sigma2))

def contiguous_clusters(indices: np.ndarray, max_gap: int = 8) -> list[np.ndarray]:
    indices = np.asarray(indices, dtype=int)
    if indices.size == 0:
        return []
    groups, current = [], [int(indices[0])]
    for idx in indices[1:]:
        if int(idx) - current[-1] <= max_gap:
            current.append(int(idx))
        else:
            groups.append(np.asarray(current, dtype=int))
            current = [int(idx)]
    groups.append(np.asarray(current, dtype=int))
    return groups


def strongest_cluster(z: np.ndarray, relative_threshold: float = 0.65, max_gap: int = 8) -> np.ndarray:
    if z.size == 0 or not np.isfinite(z).all():
        raise ValueError("Amplitude vector must be finite and non-empty")
    candidates = np.flatnonzero(z >= relative_threshold * np.max(z))
    if candidates.size == 0:
        return np.asarray([int(np.argmax(z))])
    groups = contiguous_clusters(candidates, max_gap=max_gap)
    return max(groups, key=lambda g: float(np.max(z[g])))


def centered_candidate_indices(center: int, k: int, size: int) -> np.ndarray:
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    half = k // 2
    return np.unique(np.clip(np.arange(center - half, center + half + 1), 0, size - 1))


def main_lobe_width(
    z: np.ndarray,
    u_grid: np.ndarray,
    center_idx: int,
    threshold: float,
) -> tuple[float, tuple[int, int]]:
    center_idx = int(np.clip(center_idx, 0, len(z) - 1))
    denom = max(float(z[center_idx]), np.finfo(float).tiny)
    normalized = z / denom
    left = center_idx
    right = center_idx
    while left > 0 and normalized[left - 1] >= threshold:
        left -= 1
    while right < len(z) - 1 and normalized[right + 1] >= threshold:
        right += 1
    grid_step = float(np.median(np.diff(u_grid)))
    width = max(float(u_grid[right] - u_grid[left] + grid_step), grid_step)
    return width, (left, right)


def candidate_angle_indices(z: np.ndarray, u_grid: np.ndarray, k: int, gap: int = 8) -> np.ndarray:
    cluster = strongest_cluster(z, relative_threshold=0.65, max_gap=gap)
    midpoint = int(np.rint(np.median(cluster)))
    return centered_candidate_indices(midpoint, k, len(u_grid))


def select_with_refinement_beams(
    cfg: SimulationConfig,
    h: np.ndarray,
    candidates: list[tuple[float, float]],
    sigma2: float,
    rng: np.random.Generator,
) -> int:
    beams = np.vstack([near_field_steering(cfg, u, r) for u, r in candidates])
    scores = amplitude_measurements(h, beams, sigma2, rng, path_gain(cfg, candidates[0][1]))
    return int(np.argmax(scores))

def coarse_candidates(
    cfg: SimulationConfig,
    z_dft: np.ndarray,
    u_grid: np.ndarray,
    k: int = 3,
    rho: float = 0.5,
) -> list[dict[str, float]]:
    indices = candidate_angle_indices(z_dft, u_grid, k=k)
    results = []
    for idx in indices:
        u_hat = float(u_grid[idx])
        width, bounds = main_lobe_width(z_dft, u_grid, int(idx), threshold=rho)
        r_hat = cfg.N * cfg.spacing * (1.0 - u_hat**2) / max(width, 1e-12)
        r_hat = float(np.clip(r_hat, cfg.r_min, cfg.r_max))
        results.append({"u": u_hat, "r": r_hat, "width": width, "index": int(idx), "bounds": bounds})
    return results


def estimate_coarse(
    cfg: SimulationConfig,
    h: np.ndarray,
    z_dft: np.ndarray,
    u_grid: np.ndarray,
    sigma2: float,
    rng: np.random.Generator,
    measurement_gain: float,
    k: int = 3,
) -> Estimate:
    candidates = coarse_candidates(cfg, z_dft, u_grid, k=k)
    beams = np.vstack([near_field_steering(cfg, c["u"], c["r"]) for c in candidates])
    scores = amplitude_measurements(h, beams, sigma2, rng, measurement_gain)
    chosen = candidates[int(np.argmax(scores))]
    return Estimate(
        "Coarse", chosen["u"], chosen["r"], len(u_grid) + len(candidates),
        {"candidate_count": len(candidates), "width": chosen["width"], "bounds": chosen["bounds"]},
    )


def exact_half_power_threshold(alpha: float) -> float:
    '''Paper Eq. (31) specialized to rho=0.5 (s_rho=0).'''
    alpha = max(float(alpha), 1e-10)
    factor = np.exp(1j * 3.0 * np.pi / 4.0) * np.sqrt(np.pi * alpha)
    denominator = erf(factor)
    if abs(denominator) < 1e-12:
        return 0.5
    value = 0.5 * abs(erf(2.0 * factor) / denominator)
    return float(np.clip(value, 0.05, 0.95))


def estimate_refined(
    cfg: SimulationConfig,
    h: np.ndarray,
    z_dft: np.ndarray,
    u_grid: np.ndarray,
    sigma2: float,
    rng: np.random.Generator,
    measurement_gain: float,
    k: int = 3,
    max_iterations: int = 6,
) -> Estimate:
    candidates = coarse_candidates(cfg, z_dft, u_grid, k=k)
    refined = []
    grid_step = float(np.median(np.diff(u_grid)))
    for item in candidates:
        r_current = item["r"]
        previous_width = item["width"]
        trace = []
        for iteration in range(max_iterations):
            alpha = cfg.N**2 * cfg.spacing * (1.0 - item["u"]**2) / (8.0 * max(r_current, 1e-12))
            threshold = exact_half_power_threshold(alpha)
            width, bounds = main_lobe_width(z_dft, u_grid, item["index"], threshold)
            r_new = cfg.N * cfg.spacing * (1.0 - item["u"]**2) / max(width, 1e-12)
            r_new = float(np.clip(r_new, cfg.r_min, cfg.r_max))
            trace.append((iteration + 1, alpha, threshold, width, r_new))
            r_current = r_new
            if abs(width - previous_width) < 0.5 * grid_step:
                break
            previous_width = width
        refined.append({**item, "r": r_current, "trace": trace, "width": previous_width})

    beams = np.vstack([near_field_steering(cfg, c["u"], c["r"]) for c in refined])
    scores = amplitude_measurements(h, beams, sigma2, rng, measurement_gain)
    chosen = refined[int(np.argmax(scores))]
    return Estimate(
        "Refined", chosen["u"], chosen["r"], len(u_grid) + len(refined),
        {"candidate_count": len(refined), "iterations": len(chosen["trace"]), "trace": chosen["trace"]},
    )

def log_i0(x: np.ndarray) -> np.ndarray:
    return np.log(i0e(x) + np.finfo(float).tiny) + np.abs(x)


def rice_nll(
    params: np.ndarray,
    z: np.ndarray,
    cfg: SimulationConfig,
    W_dft: np.ndarray,
) -> float:
    u, r, log_D, log_sigma2 = map(float, params)
    if not (-0.999 <= u <= 0.999 and cfg.r_min <= r <= cfg.r_max):
        return 1e30
    D = np.exp(np.clip(log_D, -30.0, 30.0))
    sigma2 = np.exp(np.clip(log_sigma2, -40.0, 20.0))
    pattern = predicted_dft_amplitudes(cfg, u, r, W_dft)
    signal_amplitude = D * pattern / max(r, 1e-12)
    z_safe = np.maximum(z, np.finfo(float).tiny)
    argument = 2.0 * z_safe * signal_amplitude / sigma2
    log_pdf = (
        np.log(2.0 * z_safe) - np.log(sigma2)
        - (z_safe**2 + signal_amplitude**2) / sigma2
        + log_i0(argument)
    )
    value = -float(np.sum(log_pdf))
    return value if np.isfinite(value) else 1e30


def clip_params(params: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(params, bounds[:, 0]), bounds[:, 1])


def finite_difference_gradient(
    objective: Callable[[np.ndarray], float],
    params: np.ndarray,
    steps: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=float)
    for i, step in enumerate(steps):
        plus, minus = params.copy(), params.copy()
        plus[i] += step
        minus[i] -= step
        gradient[i] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def simulated_annealing(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    bounds: np.ndarray,
    rng: np.random.Generator,
    iterations: int = 40,
) -> tuple[np.ndarray, list[float]]:
    current = clip_params(initial.copy(), bounds)
    current_value = objective(current)
    best, best_value = current.copy(), current_value
    scale = np.array([0.04, 0.25, 0.15, 0.15])
    history = [best_value]
    for iteration in range(iterations):
        temperature = max(1e-4, 0.5 * 0.94**iteration)
        proposal = clip_params(current + rng.normal(size=4) * scale * temperature, bounds)
        proposal_value = objective(proposal)
        delta = proposal_value - current_value
        if delta <= 0 or rng.random() < np.exp(-min(delta / temperature, 700.0)):
            current, current_value = proposal, proposal_value
        if current_value < best_value:
            best, best_value = current.copy(), current_value
        history.append(best_value)
    return best, history


def adam_refine(
    objective: Callable[[np.ndarray], float],
    initial: np.ndarray,
    bounds: np.ndarray,
    iterations: int = 70,
) -> tuple[np.ndarray, list[float]]:
    params = initial.copy()
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    finite_steps = np.array([2e-4, 2e-3, 2e-3, 2e-3])
    learning_rate = np.array([2e-3, 2e-2, 8e-3, 8e-3])
    history = [objective(params)]
    best, best_value = params.copy(), history[-1]
    for iteration in range(1, iterations + 1):
        gradient = finite_difference_gradient(objective, params, finite_steps)
        gradient = np.clip(gradient, -1e4, 1e4)
        m = beta1 * m + (1.0 - beta1) * gradient
        v = beta2 * v + (1.0 - beta2) * gradient**2
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        params = clip_params(params - learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8), bounds)
        value = objective(params)
        if value < best_value:
            best, best_value = params.copy(), value
        history.append(best_value)
    return best, history


def mle_initialization(
    coarse: Estimate,
    z: np.ndarray,
    cfg: SimulationConfig,
    W_dft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pattern = predicted_dft_amplitudes(cfg, coarse.u_hat, coarse.r_hat, W_dft) / coarse.r_hat
    strong = pattern >= 0.5 * np.max(pattern)
    weak = pattern <= 0.15 * np.max(pattern)
    D_hat = float(np.sum(z[strong] * pattern[strong]) / max(np.sum(pattern[strong] ** 2), 1e-15))
    sigma2_hat = float(np.mean(z[weak] ** 2)) if np.any(weak) else float(np.percentile(z**2, 20))
    D_hat = max(D_hat, 1e-8)
    sigma2_hat = max(sigma2_hat, 1e-14)
    initial = np.array([coarse.u_hat, coarse.r_hat, np.log(D_hat), np.log(sigma2_hat)])
    bounds = np.array([
        [-0.999, 0.999],
        [cfg.r_min, cfg.r_max],
        [np.log(D_hat) - 4.0, np.log(D_hat) + 4.0],
        [np.log(sigma2_hat) - 4.0, np.log(sigma2_hat) + 4.0],
    ])
    return initial, bounds


def estimate_mle(
    cfg: SimulationConfig,
    z_dft: np.ndarray,
    W_dft: np.ndarray,
    coarse: Estimate,
    rng: np.random.Generator,
    sa_iterations: int = 40,
    adam_iterations: int = 70,
) -> Estimate:
    initial, bounds = mle_initialization(coarse, z_dft, cfg, W_dft)
    objective = lambda p: rice_nll(p, z_dft, cfg, W_dft)
    sa_params, sa_history = simulated_annealing(objective, initial, bounds, rng, sa_iterations)
    mle_params, adam_history = adam_refine(objective, sa_params, bounds, adam_iterations)
    return Estimate(
        "MLE", float(mle_params[0]), float(mle_params[1]), len(z_dft),
        {
            "D_hat": float(np.exp(mle_params[2])),
            "sigma2_hat": float(np.exp(mle_params[3])),
            "initial": initial,
            "sa_history": sa_history,
            "adam_history": adam_history,
            "nll": float(objective(mle_params)),
        },
    )


def observed_crb(
    cfg: SimulationConfig,
    z_dft: np.ndarray,
    W_dft: np.ndarray,
    estimate: Estimate,
) -> tuple[np.ndarray, np.ndarray]:
    D_hat = float(estimate.details.get("D_hat", cfg.amplitude_factor))
    sigma2_hat = float(estimate.details.get("sigma2_hat", np.var(z_dft)))
    params = np.array([estimate.u_hat, estimate.r_hat, np.log(D_hat), np.log(max(sigma2_hat, 1e-15))])
    steps = np.array([2e-4, 2e-3, 2e-3, 2e-3])

    def per_sample_log_likelihood(p: np.ndarray) -> np.ndarray:
        u, r, log_D, log_sigma2 = map(float, p)
        D, sigma2 = np.exp(log_D), np.exp(log_sigma2)
        pattern = predicted_dft_amplitudes(cfg, u, r, W_dft)
        mu = D * pattern / max(r, 1e-12)
        z = np.maximum(z_dft, np.finfo(float).tiny)
        x = 2.0 * z * mu / sigma2
        return np.log(2.0 * z) - np.log(sigma2) - (z**2 + mu**2) / sigma2 + log_i0(x)

    scores = np.empty((len(z_dft), 4))
    for j, step in enumerate(steps):
        plus, minus = params.copy(), params.copy()
        plus[j] += step
        minus[j] -= step
        scores[:, j] = (per_sample_log_likelihood(plus) - per_sample_log_likelihood(minus)) / (2.0 * step)
    fim = scores.T @ scores
    crb = np.linalg.pinv(fim, rcond=1e-10)
    return fim, crb


def estimate_full_csi(true_u: float, true_r: float) -> Estimate:
    return Estimate("Full CSI", float(true_u), float(true_r), 0, {"oracle": True})