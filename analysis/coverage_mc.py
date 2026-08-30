"""
Does the coverage result survive a real estimator?

The coverage claim - 64 wideband pilots matching the 256-pilot narrowband
sweep - rests on the CRB. The threshold claim rested on similarly limited
evidence and collapsed once it was checked properly, so coverage gets the same
scrutiny here: the actual profiled amplitude least-squares estimator, over the
whole (theta, r) plane, on the energy-referenced axis.

Conventions, all three chosen to disfavour the claim rather than flatter it:

  * every scheme runs at the SAME A0, with A = A0/sqrt(M). Per-pilot energy is
    therefore fixed, so pilot count is total energy: the 64-pilot schemes get a
    quarter of the energy the 256-pilot one does.
  * success is the realised error, |r_hat - r|/r <= 20%, not a bound. This
    conflates precision with outliers on purpose - it is the end-to-end
    quantity a beam training system actually cares about.
  * the codebook is a fixed decimation of the DFT grid over the whole sector.
    The user falls where it falls; nothing is centred on the truth.

The comparison to beat is the CRB table in FINDINGS.md, which this reproduces
cell for cell:

    r (m)    NB-256   NB-64   WB-64
        6      100%     96%    100%
        8      100%     88%    100%
       12      100%     68%    100%
       18      100%     36%    100%
       25       92%      8%     88%
       35       92%      0%     56%

Run:  python3 analysis/coverage_mc.py
"""
import numpy as np

from wideband_ps_dft_identifiability import Array
from estimator import steering_gain, build_dictionary, estimate, _scheme, R_MIN, R_MAX

TOL = 0.20          # success: |r_hat - r| / r <= TOL
A0 = 30.0           # same operating point as the CRB table
N_TRIALS = 100
N_ANGLES = 13
RANGES = (6., 8., 12., 18., 25., 35.)

SCHEMES = (("NB-256  full codebook", 1, None),
           ("NB-64   every 4th beam", 4, None),
           ("WB-64   every 4th beam", 4, 0.05))


def coverage(arr, dec, beta, r0, th_grid, u_grid, D, phis, fm, A, rng_seed=31):
    """fraction of (angle, noise) trials landing within TOL of the true range"""
    ok = 0
    total = 0
    for theta in np.linspace(-0.85, 0.85, N_ANGLES):
        y0 = A * steering_gain(arr, theta, r0, fm, phis)
        rng = np.random.default_rng(rng_seed)
        for _ in range(N_TRIALS):
            n = np.sqrt(0.5) * (rng.standard_normal(y0.shape)
                                + 1j * rng.standard_normal(y0.shape))
            _, r_hat = estimate(arr, np.abs(y0 + n), fm, phis, th_grid, u_grid, D)
            ok += abs(r_hat - r0) / r0 <= TOL
            total += 1
    return ok / total


def main():
    arr = Array(N=256, fc=40e9)
    th_grid = np.linspace(-0.9, 0.9, 121)
    u_grid = 1.0 / np.linspace(R_MIN, R_MAX, 45)

    print("Estimator-based coverage over the (theta, r) plane.")
    print(f"Profiled amplitude least squares, {N_TRIALS} trials at each of "
          f"{N_ANGLES} angles.")
    print(f"All schemes at A0={A0:.0f} with A = A0/sqrt(M): per-pilot energy fixed,")
    print(f"so the 64-pilot schemes use a quarter of the 256-pilot one's energy.")
    print(f"success = |r_hat - r|/r <= {TOL:.0%}\n")

    built = []
    for name, dec, b in SCHEMES:
        phis, fm, A = _scheme(arr, dec, b, A0)
        built.append((name, phis, fm, A,
                      build_dictionary(arr, fm, phis, th_grid, u_grid)))

    print(f"    {'r (m)':>6} | " + " | ".join(f"{n:>22}" for n, *_ in built))
    for r0 in RANGES:
        cells = []
        for name, phis, fm, A, D in built:
            cells.append(coverage(arr, None, None, r0, th_grid, u_grid, D, phis, fm, A))
        print(f"    {r0:6.0f} | " + " | ".join(f"{c:>21.0%} " for c in cells))

    print("\n  Compare cell for cell with the CRB table in the docstring. A drop of a")
    print("  few points is the estimator's inefficiency; a collapse means the CRB was")
    print("  describing a bound the estimator cannot reach, which is how the threshold")
    print("  claim failed.")


if __name__ == "__main__":
    main()
