"""
Two gaps between the N=128 demonstration and something publishable:
the authors' own N=512 configuration, and behaviour across SNR.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wang_full as w
from cheap_init import windowed_init


def sweep(cfg, pts, snrs, n_seeds, label):
    u_grid, W_dft = w.dft_codebook(cfg)
    print(f"\n=== {label} ===")
    print(f"  N={cfg.N}, f_c={cfg.fc/1e9:.0f} GHz, r in [{cfg.r_min},{cfg.r_max}] m, "
          f"{n_seeds} seeds")
    print(f"  {'SNR':>4} {'u':>5} {'r':>6} {'Wbin':>6} | {'coarse':>8} {'MLE theirs':>11}"
          f" {'MLE windowed':>13} | {'gain':>7}", flush=True)
    for snr in snrs:
        agg = np.zeros(3)
        for (u, r) in pts:
            wb = cfg.N ** 2 * cfg.spacing * (1 - u ** 2) / (2 * r)
            e = {k: [] for k in "ctw"}
            for s in range(n_seeds):
                rng = np.random.default_rng(400 + s)
                s2 = w.reference_noise_variance(cfg, snr)
                h = w.near_field_steering(cfg, u, r)
                gn = w.path_gain(cfg, r)
                z = w.amplitude_measurements(h, W_dft, s2, rng, gn)
                c = w.estimate_coarse(cfg, h, z, u_grid, s2, rng, gn)
                iw = windowed_init(cfg, z, W_dft, c.u_hat)
                mt = w.estimate_mle(cfg, z, W_dft, c, np.random.default_rng(900 + s), 40, 70)
                mw = w.estimate_mle(cfg, z, W_dft, iw, np.random.default_rng(900 + s), 40, 70)
                e["c"].append(abs(c.r_hat - r) / r)
                e["t"].append(abs(mt.r_hat - r) / r)
                e["w"].append(abs(mw.r_hat - r) / r)
            m = [100 * np.median(e[k]) for k in "ctw"]
            agg += np.array(m)
            print(f"  {snr:4.0f} {u:5.2f} {r:6.1f} {wb:6.1f} | {m[0]:7.1f}% {m[1]:10.1f}%"
                  f" {m[2]:12.1f}% | {m[1]/max(m[2],1e-9):6.1f}x", flush=True)
        a = agg / len(pts)
        print(f"  {snr:4.0f} {'mean':>18} | {a[0]:7.1f}% {a[1]:10.1f}% {a[2]:12.1f}%"
              f" | {a[1]/max(a[2],1e-9):6.1f}x\n", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("snr", "both"):
        sweep(w.QUICK_CFG,
              ((0.0, 2.), (0.3, 3.), (0.0, 5.), (0.6, 4.), (0.0, 10.)),
              (0, 5, 10, 20, 30), 12, "SNR sweep, N=128")
    if which in ("paper", "both"):
        sweep(w.PAPER_CFG,
              ((0.0, 20.), (0.3, 40.), (0.6, 50.), (0.0, 60.), (0.3, 80.)),
              (20, 30), 8, "Authors' PAPER_CFG, N=512")
