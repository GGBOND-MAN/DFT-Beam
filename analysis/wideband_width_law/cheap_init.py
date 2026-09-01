"""
Can the initialisation be fixed without paying for a 2-D search?

The angle was never the broken part: Wang 2026's coarse angle error is 0.0047
at both 0 and 30 dB, i.e. codebook-grid quantisation rather than noise. Only
the RANGE is quantisation-broken, because it is read from a thresholded main
lobe that spans N^2 d (1-u^2) / (2r) bins and that number falls with range.

So hold their coarse angle and scan only r. That costs n_r objective
evaluations - fewer than their SA stage - instead of the n_u * n_r of a full
2-D grid.

Four schemes, all sharing their objective, SA and Adam; only the starting point
differs, so the comparison isolates initialisation.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wang_full as w
from global_init import global_initial_estimate

N_R = 120


def range_scan_init(cfg, z, W_dft, u_fixed, n_r=N_R):
    """1-D profiled amplitude LS over 1/r at the coarse angle. n_r evaluations.
    Fails at broadside: their grid u = -1 + (2k+1)/K excludes u=0, so the coarse
    angle is off by half a bin there and freezing it poisons the range scan."""
    inv_r = np.linspace(1.0 / cfg.r_max, 1.0 / cfg.r_min, n_r)
    best = (-np.inf, cfg.r_min)
    for iv in inv_r:
        r = 1.0 / iv
        g = w.predicted_dft_amplitudes(cfg, u_fixed, r, W_dft)
        s = float((z * g).sum() ** 2 / max((g * g).sum(), 1e-30))
        if s > best[0]:
            best = (s, r)
    return w.Estimate("RangeScan", float(u_fixed), float(best[1]), len(z), {})


def windowed_init(cfg, z, W_dft, u_coarse, n_r=N_R, n_u=7, half_bins=1.5):
    """Global over r, local over u: a window of +-half_bins codebook bins around
    the coarse angle. n_u * n_r evaluations - the half-bin quantisation of the
    coarse angle is enough to break a frozen-angle scan, so u cannot be fixed,
    but it does not have to be searched globally either."""
    step = 2.0 / (cfg.N * cfg.dft_oversampling)
    us = u_coarse + np.linspace(-half_bins, half_bins, n_u) * step
    inv_r = np.linspace(1.0 / cfg.r_max, 1.0 / cfg.r_min, n_r)
    best = (-np.inf, u_coarse, cfg.r_min)
    for u in us:
        for iv in inv_r:
            r = 1.0 / iv
            g = w.predicted_dft_amplitudes(cfg, float(np.clip(u, -0.95, 0.95)), r, W_dft)
            s = float((z * g).sum() ** 2 / max((g * g).sum(), 1e-30))
            if s > best[0]:
                best = (s, u, r)
    return w.Estimate("Windowed", float(np.clip(best[1], -0.95, 0.95)),
                      float(best[2]), len(z), {})


def compare(cfg=None, snr_db=30, n_seeds=12):
    cfg = cfg or w.QUICK_CFG
    u_grid, W_dft = w.dft_codebook(cfg)
    pts = ((0.0, 2.), (0.3, 3.), (0.0, 5.), (0.6, 4.), (0.3, 8.), (0.0, 10.), (0.6, 10.))
    print("Range error by initialisation. Their objective / SA / Adam unchanged.")
    print(f"  N={cfg.N}, SNR={snr_db} dB, {n_seeds} seeds\n")
    print(f"  {'u':>4} {'r':>6} {'Wbin':>6} | {'coarse':>8} {'theirs':>8}"
          f" {'2-D':>8} {'1-D':>8} {'windowed':>9} | {'win gain':>9}")
    tot = np.zeros(5)
    for (u, r) in pts:
        wb = cfg.N ** 2 * cfg.spacing * (1 - u ** 2) / (2 * r)
        e = {k: [] for k in "ctgsw"}
        for s in range(n_seeds):
            rng = np.random.default_rng(400 + s)
            s2 = w.reference_noise_variance(cfg, snr_db)
            h = w.near_field_steering(cfg, u, r)
            gn = w.path_gain(cfg, r)
            z = w.amplitude_measurements(h, W_dft, s2, rng, gn)
            c = w.estimate_coarse(cfg, h, z, u_grid, s2, rng, gn)
            i2 = global_initial_estimate(cfg, z, W_dft)
            i1 = range_scan_init(cfg, z, W_dft, c.u_hat)
            iw = windowed_init(cfg, z, W_dft, c.u_hat)
            mk = lambda init: w.estimate_mle(cfg, z, W_dft, init,
                                             np.random.default_rng(900 + s), 40, 70)
            e["c"].append(abs(c.r_hat - r) / r)
            e["t"].append(abs(mk(c).r_hat - r) / r)
            e["g"].append(abs(mk(i2).r_hat - r) / r)
            e["s"].append(abs(mk(i1).r_hat - r) / r)
            e["w"].append(abs(mk(iw).r_hat - r) / r)
        m = [100 * np.median(e[k]) for k in "ctgsw"]
        tot += np.array(m)
        print(f"  {u:4.1f} {r:6.1f} {wb:6.1f} | {m[0]:7.1f}% {m[1]:7.1f}%"
              f" {m[2]:7.1f}% {m[3]:7.1f}% {m[4]:8.1f}% | {m[1]/max(m[4],1e-9):8.1f}x")
    a = tot / len(pts)
    print(f"\n  {'mean':>18} | {a[0]:7.1f}% {a[1]:7.1f}% {a[2]:7.1f}% {a[3]:7.1f}% {a[4]:8.1f}%")
    n_u = 181
    print(f"\n  initialisation cost, objective evaluations:")
    print(f"    their coarse : ~1        (closed form, plus the N-beam sweep they already pay for)")
    print(f"    1-D r scan   : {N_R}      (fewer than their SA stage, which is 40 + Adam's ~630)")
    print(f"    windowed     : {7*N_R}      (7 angle bins x {N_R} ranges, near their Adam stage)")
    print(f"    2-D grid     : {n_u*N_R}   ({n_u*N_R/(7*N_R):.0f}x the windowed scan)")


if __name__ == "__main__":
    compare()
