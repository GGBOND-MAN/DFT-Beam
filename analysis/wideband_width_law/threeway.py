"""
The objection this has to answer: why sum the subcarriers at all?

An OFDM receiver gets a per-subcarrier amplitude for free, and the centre
subcarrier alone carries no wideband bias - Wang 2026 then applies unchanged.
Summing is only worth doing if it buys enough SNR to beat that, which is what
this measures.

All three schemes see the SAME OFDM waveform: total transmit power fixed and
spread equally over M subcarriers, so the per-subcarrier amplitude is
A0/sqrt(M). They differ only in how the receiver processes it.

  single      use the centre subcarrier's amplitude, discard the other M-1
  summed      sum the subcarrier powers, keep all the energy, accept the bias
  corrected   same measurement, then Wc = (W_meas - beta*|u|) / B(beta)

B(beta) was calibrated noiselessly on a separate 9-point training set; the
grid below is the held-out set. Range error is scored against the NARROWBAND
estimate at the same (u,r), so Wang's own narrowband error cancels and only
the wideband effect is measured.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from wang import (PAPER_CFG, element_offsets, dft_codebook, main_lobe_width,
                  candidate_angle_indices, C0)

cfg = PAPER_CFG
u_grid, W = dft_codebook(cfg)
Y = element_offsets(cfg)
B_CAL = {0.05: 0.734, 0.10: 0.633, 0.20: 0.550, 0.30: 0.488}
TEST = [(u, r) for u in (0.15, 0.30, 0.45, 0.55, 0.70) for r in (8., 14., 22.)]


def clean(u, r, f):
    rn = np.sqrt(r ** 2 + Y ** 2 - 2 * r * u * Y)
    return W @ np.conjugate(np.exp(-1j * 2 * np.pi * (rn - r) * f / C0) / np.sqrt(cfg.N))


def width_of(z):
    idx = candidate_angle_indices(z, u_grid, k=3)
    c = int(np.rint(np.median(idx)))
    w, _ = main_lobe_width(z, u_grid, c, threshold=0.5)
    return w, u_grid[c]


def range_of(w, u):
    """Wang 2026's formula INCLUDING their clip to [r_min, r_max]. Omitting the
    clip lets a noise realisation that drives the corrected width negative
    produce a 1e9 m estimate, which then dominates any average."""
    r = cfg.N * cfg.spacing * (1 - u ** 2) / max(w, 1e-12)
    return float(np.clip(r, cfg.r_min, cfg.r_max))


def run(beta, M, snr_db, n_trials=200, seed=7):
    """returns mean |relative range error| for the three schemes"""
    fm = cfg.fc + beta * cfg.fc / M * (np.arange(1, M + 1) - 1 - (M - 1) / 2)
    mid = M // 2
    B = B_CAL[beta]
    out = np.zeros(3)
    for (u, r) in TEST:
        S = np.stack([clean(u, r, f) for f in fm])            # M x K, unit amplitude
        # reference: what the narrowband method returns at this (u,r), noiseless
        w0, u0 = width_of(np.abs(S[mid]))
        r_ref = range_of(w0, u0)
        # total power fixed and spread over M subcarriers; snr_db is defined on
        # the PEAK of the combined measurement so the axis means something
        peak = np.abs(S).max()
        A = np.sqrt(M) * np.sqrt(10 ** (snr_db / 10.0)) / peak
        rng = np.random.default_rng(seed)
        acc = np.zeros(3)
        for _ in range(n_trials):
            n = np.sqrt(0.5) * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
            Z = np.abs(A * S + n)
            w_s, u_s = width_of(Z[mid])                        # single subcarrier
            w_m, u_m = width_of(np.sqrt((Z ** 2).mean(0)))     # summed
            acc[0] += abs(range_of(w_s, u_s) - r_ref) / r_ref
            acc[1] += abs(range_of(w_m, u_m) - r_ref) / r_ref
            acc[2] += abs(range_of(max((w_m - beta * abs(u_m)) / B, 1e-9), u_m) - r_ref) / r_ref
        out += acc / n_trials
    return 100 * out / len(TEST)


if __name__ == "__main__":
    M = 9
    print(f"Three-way comparison. {len(TEST)} held-out points, 200 noise trials each,")
    print(f"M={M} subcarriers, total transmit power fixed across all schemes.")
    print("Error is relative to the narrowband estimate at the same (u,r).\n")
    for beta in (0.10, 0.20):
        print(f"  B/fc = {beta}   (B = {B_CAL[beta]})")
        print(f"    {'SNR dB':>7} | {'single':>9} {'summed':>9} {'corrected':>11} | {'winner':>10}")
        for snr in (0, 5, 10, 15, 20, 30):
            e = run(beta, M, snr)
            win = ("single", "summed", "corrected")[int(np.argmin(e))]
            print(f"    {snr:7.0f} | {e[0]:8.1f}% {e[1]:8.1f}% {e[2]:10.1f}% | {win:>10}")
        print()
