"""
Dither-matched codebook shaping for TTD-free wideband near-field beam training.

Uniform decimation of the DFT grid spends pilots badly: the frequency dither
Delta(theta) = beta*N*|theta|/2 (in DFT-beam units) vanishes at broadside and
grows with |theta|, so a uniform codebook is starved near boresight and
over-provisioned off it. Shaping the local beam spacing s(theta) to the dither
should even that out.

Two shapes are derived and tested against uniform placement at equal pilot count:

  flat-g          s(theta) = g* + Delta(theta)
                  equalises the worst-case sampling gap g across the sector.

  width-weighted  s(theta) = g*(1-theta^2) + Delta(theta)
                  equalises g / W instead. The pattern width itself carries a
                  (1-theta^2) factor whose shape is independent of the unknown
                  range, so it is available at design time.

Run:  python3 analysis/codebook_design.py
"""
import numpy as np

from wideband_ps_dft_identifiability import Array, _crb_range

N_DEF, FC_DEF, BETA, M_SUB, THETA_MAX = 256, 40e9, 0.05, 9, 0.9


def dither_slope(N=N_DEF, beta=BETA):
    """Delta(theta) = dither_slope * |theta|, in DFT-beam units."""
    return beta * N / 2


def place(spacing, N=N_DEF, theta_max=THETA_MAX):
    """Place beams by integrating the density 1/s(theta). `spacing` returns the
    local spacing in DFT-beam units; actual angular spacing is s * 2/N."""
    g = np.linspace(0.0, theta_max, 20001)
    dens = (N / 2) / spacing(g)
    cum = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2 * np.diff(g))])
    ph = np.interp(np.arange(0, cum[-1]), cum, g)
    return np.unique(np.concatenate([-ph[::-1], ph]))


def flat_g(gstar, N=N_DEF, beta=BETA):
    c = dither_slope(N, beta)
    return lambda t: gstar + c * np.abs(t)


def width_weighted(gstar, N=N_DEF, beta=BETA):
    c = dither_slope(N, beta)
    return lambda t: gstar * (1 - t ** 2) + c * np.abs(t)


def place_with_K(shape, K_target, N=N_DEF, theta_max=THETA_MAX):
    """Bisect g* so the shaped codebook lands on K_target beams. Never truncate
    the array - dropping the outer beams uncovers the sector edge and produces
    spurious failures there."""
    lo, hi = 0.05, 200.0
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        K = len(place(shape(mid), N, theta_max))
        if K > K_target:
            lo = mid
        else:
            hi = mid
    return place(shape(hi), N, theta_max)


def snap_to_dft(phis, arr):
    """constrain a shaped codebook to be a subset of the DFT codebook"""
    return np.unique(arr.grid[np.abs(arr.grid[None, :] - phis[:, None]).argmin(1)])


def _rel_err(arr, phis, r0, fm, A, thetas):
    return np.array([_crb_range(arr, t, r0, fm, phis, A) / r0 for t in thetas])


def exp12_codebook_shapes(n_ang=101, A0=30.0):
    """Equal pilot count, wideband, three placements."""
    arr = Array(N=N_DEF, fc=FC_DEF)
    fm, A = arr.subcarriers(BETA, M_SUB), A0 / np.sqrt(M_SUB)
    thetas = np.linspace(-0.85, 0.85, n_ang)
    print(f"\n[12] Codebook shaping at equal pilot count ({n_ang} angles, wideband)")
    for r0 in (18.0, 25.0):
        print(f"    r = {r0:.0f} m")
        print(f"    {'K':>4} | {'uniform med':>11} {'p90':>7} | {'flat-g med':>10} {'p90':>7}"
              f" | {'width-wtd med':>13} {'p90':>7}")
        for gs in (5.0, 4.0, 3.0, 2.5):
            pw = place(width_weighted(gs))
            K = len(pw)
            pf = place_with_K(flat_g, K)
            pu = np.linspace(-THETA_MAX, THETA_MAX, K)
            e = [_rel_err(arr, p, r0, fm, A, thetas) for p in (pu, pf, pw)]
            print(f"    {K:4d} | " + " | ".join(
                f"{np.median(x):10.1%} {np.percentile(x, 90):7.1%}" for x in e))
        print()


def exp13_where_it_helps(K_shape_gstar=4.0, r0=18.0, n_ang=101, A0=30.0):
    """Which angles each placement wins, and what it costs elsewhere."""
    arr = Array(N=N_DEF, fc=FC_DEF)
    fm, A = arr.subcarriers(BETA, M_SUB), A0 / np.sqrt(M_SUB)
    thetas = np.linspace(-0.85, 0.85, n_ang)
    pw = place(width_weighted(K_shape_gstar))
    K = len(pw)
    pf, pu = place_with_K(flat_g, K), np.linspace(-THETA_MAX, THETA_MAX, K)
    eu, ef, ew = (_rel_err(arr, p, r0, fm, A, thetas) for p in (pu, pf, pw))
    print(f"\n[13] Median relative range error by angle (K={K}, r={r0:.0f} m)")
    print(f"    {'|theta|':>14} {'uniform':>9} {'flat-g':>9} {'width-wtd':>11}")
    for lo, hi in ((0, .15), (.15, .3), (.3, .45), (.45, .6), (.6, .75), (.75, .9)):
        s = (np.abs(thetas) >= lo) & (np.abs(thetas) < hi)
        print(f"    [{lo:.2f},{hi:.2f})".rjust(18)
              + f"{np.median(eu[s]):9.1%}{np.median(ef[s]):9.1%}{np.median(ew[s]):11.1%}")
    print("    -> shaping does move accuracy from the off-boresight angles, where the")
    print("       dither already suffices, to broadside, where it vanishes. The net")
    print("       effect over the sector is small.")


if __name__ == "__main__":
    exp12_codebook_shapes()
    exp13_where_it_helps()
