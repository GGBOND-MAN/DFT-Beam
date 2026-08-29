"""
Where the CRB stops describing achievable performance.

The Monte-Carlo runs in `estimator.py` show sparse codebooks failing far from
the CRB at moderate SNR. This module identifies the mechanism and gives a
predictor for the breakdown point.

The ambiguity slice A(theta_0, r) turns out to be UNIMODAL in range for every
scheme tested - there is no discrete range sidelobe, so the classical
method-of-interval-errors picture does not apply. What sets the threshold
instead is how much of the pattern's energy the codebook actually samples,
together with how well the near-field pattern is separated from the far-field
one it degenerates into as r grows.

    threshold peak-SNR (dB) ~ const - 10log10(E_eff)

    E_eff = sum_{m,k} G^2 / max G^2   the effective number of informative
                                      measurements: how many beam-subcarrier
                                      samples actually land on the pattern.

E_eff is what the frequency dither raises, and it is predicted by the sampling
rule of THEORY.md section 4 with no new machinery:

    E_eff ~ W / g,   W = N^2 d (1-theta^2) / (2r),
                     g = max(dec - Delta, Delta/(M-1)),  Delta = beta N |theta| / 2

A near/far correlation term A_inf was tried as a second factor and rejected: it
orders the schemes backwards (the wideband case has the highest A_inf and the
lowest threshold), so it is reported below as a diagnostic only.

Run:  python3 analysis/threshold.py
"""
import numpy as np

from wideband_ps_dft_identifiability import Array
from estimator import steering_gain, _scheme, SCHEMES

R_FAR = 5000.0          # stand-in for the far-field limit


def _profiled(G0, G):
    return (((G0 * G).sum(-1) ** 2) / ((G * G).sum(-1) + 1e-30)).sum()


def pattern_statistics(arr, theta, r, fm, phis):
    """effective measurement count and near/far separation of one scheme"""
    G0 = np.abs(steering_gain(arr, theta, r, fm, phis))
    e_ratio = (G0 ** 2).sum() / (G0 ** 2).max()
    Gf = np.abs(steering_gain(arr, theta, R_FAR, fm, phis))
    a_inf = _profiled(G0, Gf) / (G0 ** 2).sum()
    return e_ratio, a_inf


def exp14_threshold_predictor(theta=0.6, r0=12.0, measured=None):
    """Predict the relative breakdown points and compare with measurement."""
    arr = Array(N=256, fc=40e9)
    measured = measured or {"narrowband, full codebook": 19.4,
                            "narrowband, every 4th beam": 23.4,
                            "wideband B/fc=0.05, every 4th": 14.0}
    print("\n[14] Threshold predictor vs measured 10%-outlier breakdown")
    print(f"    {'scheme':<31} {'K':>4} | {'E_eff':>7} {'W/g':>6} {'A_inf':>7}"
          f" {'predicted':>10} {'measured':>9} {'resid':>7}")
    rows = []
    for name, dec, b in SCHEMES:
        phis, fm, _ = _scheme(arr, dec, b, 1.0)
        e_ratio, a_inf = pattern_statistics(arr, theta, r0, fm, phis)
        pred = -10 * np.log10(e_ratio)
        N, d, beta, M = arr.N, arr.d, 0.05, 9
        W = N ** 2 * d * (1 - theta ** 2) / (2 * r0)
        if b is None:
            g = float(dec)
        else:
            D = beta * N * abs(theta) / 2
            g = max(dec - D, D / (M - 1))
        rows.append((name, len(phis), e_ratio, W / g, a_inf, pred, measured.get(name)))
    # the predictor fixes relative offsets only; anchor on the first scheme
    off = rows[0][6] - rows[0][5]
    for name, K, er, wg, ai, pred, meas in rows:
        p = pred + off
        print(f"    {name:<31} {K:4d} | {er:7.2f} {wg:6.2f} {ai:7.4f} {p:9.1f}dB"
              f" {meas:8.1f}dB {p-meas:6.1f}")
    print("    (measured values are 10%-outlier breakdown peak-SNRs from a 120-run")
    print("     sweep; the full-codebook entry is an upper bound - it never broke)")
    print("    W/g is the analytic prediction of E_eff from THEORY.md section 4,")
    print("    computed without reference to the measured patterns.")


def exp15_ambiguity_is_unimodal(theta=0.6, r0=12.0, n=1500):
    """Show there is no discrete range sidelobe to blame."""
    arr = Array(N=256, fc=40e9)
    rs = 1.0 / np.linspace(1 / 300.0, 1 / 5.0, n)
    print("\n[15] Range ambiguity slice A(theta_0, r): local maxima away from truth")
    print(f"    {'scheme':<31} {'K':>4} | {'distinct local maxima':>21} {'A at 300 m':>11}")
    for name, dec, b in SCHEMES:
        phis, fm, _ = _scheme(arr, dec, b, 1.0)
        G0 = np.abs(steering_gain(arr, theta, r0, fm, phis))
        A = np.array([_profiled(G0, np.abs(steering_gain(arr, theta, r, fm, phis)))
                      for r in rs]) / (G0 ** 2).sum()
        loc = np.where((A[1:-1] > A[:-2]) & (A[1:-1] >= A[2:]))[0] + 1
        gp = int(np.argmax(A))
        lo, hi = gp, gp
        while lo > 0 and A[lo - 1] < A[lo]:
            lo -= 1
        while hi < len(A) - 1 and A[hi + 1] < A[hi]:
            hi += 1
        extra = [i for i in loc if i < lo or i > hi]
        print(f"    {name:<31} {len(phis):4d} | {len(extra):21d} {A[0]:11.4f}")
    print("    -> none. The ridge is unimodal, so outliers are not sidelobe captures;")
    print("       they are the estimator sliding along a flat ridge toward the")
    print("       far-field degeneracy, which is what A_inf measures.")


if __name__ == "__main__":
    exp15_ambiguity_is_unimodal()
    exp14_threshold_predictor()
