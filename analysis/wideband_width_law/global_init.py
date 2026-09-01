"""
One-function improvement to Wang 2026: replace the MLE's initialisation.

Their MLE maximises the exact Rician likelihood but starts from the coarse
width-based estimate and searches locally (SA proposal scale [0.04, 0.25, ...]
times a temperature that starts at 0.5 and cools as 0.94^k). The coarse range
is read from a THRESHOLDED main-lobe width, so it is quantised to the codebook
grid, and the quantisation gets worse as the lobe narrows with range:

    W_bins = N^2 d (1-u^2) / (2r)

Their own numbers track that quantity and nothing else - 2.4% error at 6 bins,
96% at 2 bins. Below a few bins the MLE inherits the failure instead of fixing
it, because it cannot travel far enough to escape.

This swaps in a global initialisation - a coarse grid over (u, 1/r) scored by
profiled amplitude least squares - and leaves their objective, their SA and
their Adam untouched, so any difference is attributable to the initialisation
alone.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wang_full as w


def global_initial_estimate(cfg, z, W_dft, n_u=181, n_r=120):
    """profiled amplitude LS over a grid in (u, 1/r); the per-beam gain is
    eliminated in closed form so no amplitude has to be guessed"""
    us = np.linspace(-0.95, 0.95, n_u)
    inv_r = np.linspace(1.0 / cfg.r_max, 1.0 / cfg.r_min, n_r)
    best = (-np.inf, 0.0, cfg.r_min)
    for u in us:
        for iv in inv_r:
            r = 1.0 / iv
            g = w.predicted_dft_amplitudes(cfg, u, r, W_dft)
            score = float((z * g).sum() ** 2 / max((g * g).sum(), 1e-30))
            if score > best[0]:
                best = (score, u, r)
    return w.Estimate("GlobalInit", best[1], best[2], len(z), {})


def compare(cfg=None, snr_db=30, n_seeds=12):
    cfg = cfg or w.QUICK_CFG
    u_grid, W_dft = w.dft_codebook(cfg)
    pts = ((0.0, 2.), (0.3, 3.), (0.0, 5.), (0.6, 4.), (0.3, 8.), (0.0, 10.), (0.6, 10.))
    print(f"MLE range error: their coarse initialisation vs a global one")
    print(f"  cfg N={cfg.N}, SNR={snr_db} dB, {n_seeds} seeds, their objective/SA/Adam unchanged\n")
    print(f"  {'u':>4} {'r':>6} {'W bins':>7} | {'coarse':>8} {'MLE(theirs)':>12}"
          f" {'MLE(global)':>12} | {'gain':>7}")
    for (u, r) in pts:
        wb = cfg.N ** 2 * cfg.spacing * (1 - u ** 2) / (2 * r)
        e = {k: [] for k in ("c", "t", "g")}
        for s in range(n_seeds):
            rng = np.random.default_rng(400 + s)
            s2 = w.reference_noise_variance(cfg, snr_db)
            h = w.near_field_steering(cfg, u, r)
            gn = w.path_gain(cfg, r)
            z = w.amplitude_measurements(h, W_dft, s2, rng, gn)
            c = w.estimate_coarse(cfg, h, z, u_grid, s2, rng, gn)
            gi = global_initial_estimate(cfg, z, W_dft)
            mt = w.estimate_mle(cfg, z, W_dft, c, np.random.default_rng(900 + s), 40, 70)
            mg = w.estimate_mle(cfg, z, W_dft, gi, np.random.default_rng(900 + s), 40, 70)
            e["c"].append(abs(c.r_hat - r) / r)
            e["t"].append(abs(mt.r_hat - r) / r)
            e["g"].append(abs(mg.r_hat - r) / r)
        mc, mt_, mg_ = (100 * np.median(e[k]) for k in ("c", "t", "g"))
        print(f"  {u:4.1f} {r:6.1f} {wb:7.1f} | {mc:7.1f}% {mt_:11.1f}%"
              f" {mg_:11.1f}% | {mt_/max(mg_,1e-9):6.1f}x")


if __name__ == "__main__":
    compare()
