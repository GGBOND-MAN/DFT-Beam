"""
Where does Wang 2026 actually fail? Run their own pipeline and look.

Two probes:
  1. baseline accuracy of coarse / refined / MLE across SNR and geometry
  2. whether the MLE can recover when the coarse stage is wrong

Probe 2 matters because their MLE is initialised from the coarse estimate and
searches locally: the SA proposal scale is [0.04, 0.25, 0.15, 0.15] times a
temperature starting at 0.5 and cooling as 0.94^k, so over 40 iterations the
angle wanders at most a few hundredths and the range a metre or so. If that is
right, a coarse failure is inherited rather than corrected, and the published
accuracy rests entirely on the coarse stage succeeding.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import wang_full as w

cfg = w.QUICK_CFG
u_grid, W_dft = w.dft_codebook(cfg)


def one_run(u, r, snr_db, seed, perturb=None):
    rng = np.random.default_rng(seed)
    sigma2 = w.reference_noise_variance(cfg, snr_db)
    h = w.near_field_steering(cfg, u, r)
    g = w.path_gain(cfg, r)
    z = w.amplitude_measurements(h, W_dft, sigma2, rng, g)
    coarse = w.estimate_coarse(cfg, h, z, u_grid, sigma2, rng, g)
    refined = w.estimate_refined(cfg, h, z, u_grid, sigma2, rng, g)
    init = coarse
    if perturb is not None:
        init = w.Estimate("Coarse", float(np.clip(coarse.u_hat + perturb[0], -0.95, 0.95)),
                          float(np.clip(coarse.r_hat * perturb[1], cfg.r_min, cfg.r_max)),
                          coarse.overhead, dict(coarse.details))
    mle = w.estimate_mle(cfg, z, W_dft, init, rng, sa_iterations=40, adam_iterations=70)
    return coarse, refined, mle, init


def probe1(n=12):
    print("PROBE 1 - baseline accuracy of their three schemes")
    print(f"  cfg N={cfg.N}, f_c={cfg.fc/1e9:.0f} GHz, r in [{cfg.r_min},{cfg.r_max}] m")
    print(f"  {n} (u,r) points x 8 seeds, median |dr|/r and |du|\n")
    pts = [(u, r) for u in (0.0, 0.3, 0.6) for r in (2., 4., 7., 10.)][:n]
    print(f"    {'SNR':>5} | {'coarse dr':>10} {'du':>8} | {'refined dr':>11} {'du':>8} | {'MLE dr':>8} {'du':>8}")
    for snr in (0, 10, 20, 30):
        acc = np.zeros((3, 2, 0)).tolist()
        er = [[], [], []]; ea = [[], [], []]
        for (u, r) in pts:
            for s in range(8):
                c, f, m, _ = one_run(u, r, snr, 100 + s)
                for i, e in enumerate((c, f, m)):
                    er[i].append(abs(e.r_hat - r) / r); ea[i].append(abs(e.u_hat - u))
        print(f"    {snr:5.0f} | " + " | ".join(
            f"{100*np.median(er[i]):9.1f}% {np.median(ea[i]):8.4f}" for i in range(3)))


def probe2():
    print("\nPROBE 2 - can the MLE recover from a wrong coarse estimate?")
    print("  the coarse estimate is displaced by hand, then handed to their MLE\n")
    print(f"    {'du inject':>10} {'r factor':>9} | {'init |du|':>10} {'MLE |du|':>10}"
          f" | {'init dr':>9} {'MLE dr':>9} | {'recovered?':>11}")
    for du, rf in ((0.0, 1.0), (0.02, 1.0), (0.05, 1.0), (0.10, 1.0),
                   (0.0, 1.5), (0.0, 2.5), (0.0, 0.4)):
        du_i = du_m = dr_i = dr_m = 0.0
        K = 0
        for (u, r) in ((0.3, 5.), (0.6, 5.), (0.3, 8.)):
            for s in range(6):
                c, f, m, init = one_run(u, r, 20, 200 + s, perturb=(du, rf))
                du_i += abs(init.u_hat - u); du_m += abs(m.u_hat - u)
                dr_i += abs(init.r_hat - r) / r; dr_m += abs(m.r_hat - r) / r
                K += 1
        du_i, du_m, dr_i, dr_m = du_i/K, du_m/K, dr_i/K, dr_m/K
        rec = "yes" if (dr_m < 0.5 * dr_i or dr_i < 0.05) else "NO"
        print(f"    {du:10.2f} {rf:9.1f} | {du_i:10.4f} {du_m:10.4f}"
              f" | {100*dr_i:8.1f}% {100*dr_m:8.1f}% | {rec:>11}")


if __name__ == "__main__":
    probe1()
    probe2()
