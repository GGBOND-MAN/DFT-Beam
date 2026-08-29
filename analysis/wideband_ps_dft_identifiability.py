"""
Wideband PS-only near-field DFT beam training: does the frequency dimension
carry angle-range information that the narrowband pattern does not?

Setting (ULA, LoS, single user):
  - N-element ULA, spacing d = lambda_c/2, frequency-independent phase shifters
  - conventional DFT codebook designed at f_c, reused on every OFDM subcarrier
  - amplitude-only measurements z_{m,k} = |y_{m,k}| (Rician), no carrier phase

Fairness convention used throughout:
  one OFDM symbol of fixed energy is split over M subcarriers, so the
  per-subcarrier amplitude scales as A/sqrt(M). Under this convention a pure
  "average over subcarriers" gain is exactly zero, and any CRB improvement is
  structural rather than an SNR artefact.

Run:  python3 analysis/wideband_ps_dft_identifiability.py
"""
import numpy as np
from scipy.special import i0e, i1e
from scipy.integrate import quad

C0 = 3e8


class Array:
    def __init__(self, N=512, fc=100e9):
        self.N, self.fc = N, fc
        self.d = C0 / (2 * fc)                                  # half wavelength at f_c
        self.dn = (2 * np.arange(1, N + 1) - N - 1) / 2.0
        self.grid = (2 * np.arange(1, N + 1) - N - 1) / N       # DFT codebook grid

    def gains(self, theta, r, fm, phis):
        """|b_m^H(theta,r) a(phi)| on the exact spherical wavefront. -> (M, K)"""
        rn = np.sqrt(r ** 2 + (self.dn * self.d) ** 2 - 2 * r * theta * self.dn * self.d)
        ph = np.exp(1j * 2 * np.pi * np.outer(np.atleast_1d(fm), rn - r) / C0)
        A = np.exp(-1j * np.pi * np.outer(np.atleast_1d(phis), self.dn))
        return np.abs(ph @ A.conj().T) / self.N

    def peak_beam(self, theta, r):
        g = self.gains(theta, r, self.fc, self.grid)[0]
        return self.grid[int(np.argmax(g))]

    def subcarriers(self, beta, M):
        if M == 1:
            return np.array([self.fc])
        B = beta * self.fc
        return self.fc + B / M * (np.arange(1, M + 1) - 1 - (M - 1) / 2)


def rice_info(nu, s2):
    """Fisher information about nu from one |CN(nu e^{j psi}, s2)| sample."""
    if nu < 1e-12:
        return 2.0 / s2
    s = np.sqrt(s2)

    def integrand(z):
        t = 2 * z * nu / s2
        f = (2 * z / s2) * np.exp(-(z - nu) ** 2 / s2) * i0e(t)
        return f * (4 / s2 ** 2) * (z * i1e(t) / i0e(t) - nu) ** 2

    return quad(integrand, 0.0, nu + 8 * s, limit=200)[0]


def crb_amplitude_only(arr, theta, r, fm, phis, A, s2=1.0, free_Am=False):
    """sqrt-CRB of (theta, r) from amplitude-only measurements.
    Nuisance: one common amplitude, or one amplitude per subcarrier."""
    fm, phis = np.atleast_1d(fm), np.atleast_1d(phis)
    M, K = len(fm), len(phis)
    ht, hr = 1e-6, 1e-4
    G = arr.gains(theta, r, fm, phis)
    dGt = (arr.gains(theta + ht, r, fm, phis) - arr.gains(theta - ht, r, fm, phis)) / (2 * ht)
    dGr = (arr.gains(theta, r + hr, fm, phis) - arr.gains(theta, r - hr, fm, phis)) / (2 * hr)

    Jn = np.vectorize(lambda x: rice_info(x, s2))(A * G)
    P = 2 + (M if free_Am else 1)
    D = np.zeros((M, K, P))
    D[:, :, 0], D[:, :, 1] = A * dGt, A * dGr
    if free_Am:
        for m in range(M):
            D[m, :, 2 + m] = G[m]
    else:
        D[:, :, 2] = G
    Cov = np.linalg.pinv(np.einsum('mk,mkp,mkq->pq', Jn, D, D))
    return np.sqrt(Cov[0, 0]), np.sqrt(Cov[1, 1])


def crb_coherent(arr, theta, r, fm, phis, A, s2=1.0):
    """sqrt-CRB of (theta, r) when the carrier phase IS usable (time-of-flight).
    Nuisance: complex amplitude. Local bound only - ignores phase ambiguity."""
    fm, phis = np.atleast_1d(fm), np.atleast_1d(phis)

    def sig(t, rr):
        rn = np.sqrt(rr ** 2 + (arr.dn * arr.d) ** 2 - 2 * rr * t * arr.dn * arr.d)
        ph = np.exp(-1j * 2 * np.pi * np.outer(fm, rn) / C0)     # absolute delay retained
        Ac = np.exp(-1j * np.pi * np.outer(phis, arr.dn))
        return (A * (ph @ Ac.conj().T) / arr.N).ravel()

    ht, hr = 1e-6, 1e-5
    y = sig(theta, r)
    D = np.stack([(sig(theta + ht, r) - sig(theta - ht, r)) / (2 * ht),
                  (sig(theta, r + hr) - sig(theta, r - hr)) / (2 * hr),
                  y / A, 1j * y / A], axis=1)
    Cov = np.linalg.pinv((2 / s2) * np.real(D.conj().T @ D))
    return np.sqrt(Cov[0, 0]), np.sqrt(Cov[1, 1])


# ---------------------------------------------------------------- experiments

def exp1_pattern_zooming(arr, theta=0.5, r=8.0, beta=0.1, M=5):
    """ULA form of the pattern zooming effect: centre AND width both scale by eta_m."""
    print("\n[1] Pattern zooming on a ULA (centre and width vs subcarrier)")
    fm = arr.subcarriers(beta, M)
    G = arr.gains(theta, r, fm, arr.grid)
    print(f"    true theta={theta}, r={r} m, B/fc={beta}")
    print(f"    {'f (GHz)':>9} {'eta_m':>7} {'centre':>9} {'width':>8} {'centre/eta':>11} {'width/eta':>10}")
    for i, f in enumerate(fm):
        sup = arr.grid[G[i] > 0.5 * G[i].max()]
        cen, wid, eta = sup.mean(), np.ptp(sup), f / arr.fc
        print(f"    {f/1e9:9.2f} {eta:7.4f} {cen:9.5f} {wid:8.5f} {cen/eta:11.5f} {wid/eta:10.5f}")
    print("    -> both features scale by the SAME eta_m: no new observable,")
    print("       only a known reparametrisation of the narrowband pattern.")


def exp2_dense_vs_sparse(arr, r=8.0, beta=0.2, M=9, A0=30.0):
    """Where (if anywhere) the frequency dimension actually pays off."""
    print("\n[2] Equal-energy CRB, narrowband vs wideband, dense vs sparse codebook")
    fm = arr.subcarriers(beta, M)
    print(f"    {'theta':>6} {'K':>4} {'stride':>7} {'CRBr NB':>10} {'CRBr WB':>10} {'gain':>7}")
    for theta in (0.0, 0.3, 0.5, 0.7):
        for K, stride in ((512, 1), (32, 1), (8, 4), (8, 8), (4, 8)):
            if K == arr.N:
                phis = arr.grid
            else:
                phis = arr.peak_beam(theta, r) + stride * (np.arange(K) - K // 2) * 2 / arr.N
            _, nb = crb_amplitude_only(arr, theta, r, arr.fc, phis, A0)
            _, wb = crb_amplitude_only(arr, theta, r, fm, phis, A0 / np.sqrt(M))
            print(f"    {theta:6.1f} {K:4d} {stride:7d} {nb:10.3e} {wb:10.3e} {nb/wb:6.2f}x")
        print()
    print("    -> a dense sweep gains nothing; the gain appears only for K << N,")
    print("       and shows up mainly as reduced sensitivity to where the fixed")
    print("       DFT grid happens to fall relative to the user's pattern.")


def exp3_subcarrier_count(arr, theta=0.5, r=8.0, beta=0.2, K=8, stride=8):
    """Under fixed pilot energy, more subcarriers is NOT better."""
    print("\n[3] Optimal number of subcarriers at fixed total pilot energy")
    phis = arr.peak_beam(theta, r) + stride * (np.arange(K) - K // 2) * 2 / arr.N
    for A0 in (60.0, 30.0, 15.0):
        out = []
        for M in (1, 3, 5, 9, 17, 33, 65):
            _, sr = crb_amplitude_only(arr, theta, r, arr.subcarriers(beta, M), phis, A0 / np.sqrt(M))
            out.append((M, sr))
        best = min(out, key=lambda x: x[1])[0]
        print(f"    A0={A0:5.1f}  " + "  ".join(f"M={m}:{s:.3f}" for m, s in out) + f"   best M={best}")
    print("    -> interior optimum: splitting energy over too many subcarriers")
    print("       drives each below the noncoherent (amplitude-only) knee.")


def exp4_vs_time_of_flight(arr, theta=0.5, r=8.0, K=8, stride=8, A0=30.0):
    """The competitor a reviewer will raise: cross-subcarrier phase ranging."""
    print("\n[4] Amplitude-only vs coherent time-of-flight ranging (same energy)")
    phis = arr.peak_beam(theta, r) + stride * (np.arange(K) - K // 2) * 2 / arr.N
    print(f"    {'B/fc':>6} {'M':>4} {'CRBr amp-only':>14} {'CRBr coherent':>14} {'ratio':>8}")
    for beta, M in ((0.0, 1), (0.1, 9), (0.2, 9), (0.2, 65)):
        fm = arr.subcarriers(beta, M)
        A = A0 / np.sqrt(M)
        _, amp = crb_amplitude_only(arr, theta, r, fm, phis, A)
        _, coh = crb_coherent(arr, theta, r, fm, phis, A)
        print(f"    {beta:6.2f} {M:4d} {amp:14.3e} {coh:14.3e} {amp/coh:7.1f}x")
    print("    -> coherent ranging is far tighter. Dropping phase must be justified")
    print("       by CFO / phase distortion and the absence of Tx-Rx clock sync,")
    print("       not presented as a free modelling choice.")


def exp5_cost_of_dropping_ttd(theta=0.5, r=8.0, M=65):
    """What PS-only actually costs, with perfect CSI on both sides.
    This loss is intrinsic to the front end - no estimator can recover it."""
    print("\n[5] Data-transmission ceiling: PS-only focusing vs per-subcarrier (TTD)")
    print(f"    {'N':>5} {'fc':>6} {'B/fc':>6} {'PS gain':>9} {'TTD gain':>9} {'loss dB':>8}")
    for N, fc in ((256, 40e9), (512, 100e9), (1024, 100e9)):
        arr = Array(N=N, fc=fc)
        for beta in (0.05, 0.10, 0.20):
            fm = arr.subcarriers(beta, M)
            rn = np.sqrt(r**2 + (arr.dn*arr.d)**2 - 2*r*theta*arr.dn*arr.d)
            H = np.exp(-1j*2*np.pi*np.outer(fm, rn - r)/C0)/np.sqrt(N)
            v = np.exp(-1j*2*np.pi*arr.fc*(rn - r)/C0)/np.sqrt(N)      # designed at f_c, reused
            g_ps = np.mean(np.abs(H @ v.conj())**2)*N
            g_td = N                                                    # per-subcarrier matched
            print(f"    {N:5d} {fc/1e9:5.0f}G {beta:6.2f} {g_ps:9.1f} {g_td:9.1f} "
                  f"{10*np.log10(g_td/g_ps):8.2f}")
    print("    -> the PS-only gain saturates at O(1/beta) and stops growing with N.")
    print("       An XL-array buys nothing in a wideband PS-only data link.")


def exp6_gain_vs_fractional_bandwidth(r=8.0, A0=30.0, n_angles=24):
    """Does the sparse-codebook gain need a large bandwidth? (It does not.)"""
    print("\n[6] Robustness gain vs fractional bandwidth (median / 90th pct over angles)")
    for N, fc in ((256, 40e9), (512, 100e9)):
        arr = Array(N=N, fc=fc)
        for K, stride in ((16, 4), (8, 8)):
            print(f"    N={N} fc={fc/1e9:.0f}GHz K={K} stride={stride}")
            for beta in (0.02, 0.05, 0.10, 0.20):
                pairs = []
                for th in np.linspace(-0.85, 0.85, n_angles):
                    phis = arr.peak_beam(th, r) + stride*(np.arange(K) - K//2)*2/N
                    _, nb = crb_amplitude_only(arr, th, r, arr.fc, phis, A0)
                    _, wb = crb_amplitude_only(arr, th, r, arr.subcarriers(beta, 9),
                                               phis, A0/np.sqrt(9))
                    pairs.append((nb, wb))
                nb, wb = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
                print(f"      B/fc={beta:4.2f} | median {np.median(nb):7.3f}->{np.median(wb):6.3f}"
                      f" ({np.median(nb)/np.median(wb):4.2f}x) | p90 {np.percentile(nb,90):7.3f}"
                      f"->{np.percentile(wb,90):6.3f} ({np.percentile(nb,90)/np.percentile(wb,90):5.2f}x)")
    print("    -> the gain saturates by B/fc ~ 0.02 and is driven by the tail, not the median.")
    print("       It therefore does NOT require the large fractional bandwidth at which")
    print("       PS-only data transmission becomes untenable.")


def _crb_range(arr, theta, r, fm, phis, A, s2=1.0, cond_max=1e12):
    """CRB of r alone, returning inf on a numerically singular FIM rather than
    trusting a pseudo-inverse (which silently reports a finite bound)."""
    fm, phis = np.atleast_1d(fm), np.atleast_1d(phis)
    M, K = len(fm), len(phis)
    ht, hr = 1e-6, 1e-4
    G = arr.gains(theta, r, fm, phis)
    dGt = (arr.gains(theta + ht, r, fm, phis) - arr.gains(theta - ht, r, fm, phis)) / (2 * ht)
    dGr = (arr.gains(theta, r + hr, fm, phis) - arr.gains(theta, r - hr, fm, phis)) / (2 * hr)
    Jn = np.vectorize(lambda x: rice_info(x, s2))(A * G)
    D = np.zeros((M, K, 3))
    D[:, :, 0], D[:, :, 1], D[:, :, 2] = A * dGt, A * dGr, G
    J = np.einsum('mk,mkp,mkq->pq', Jn, D, D)
    if np.linalg.cond(J) > cond_max:
        return np.inf
    return np.sqrt(np.linalg.inv(J)[1, 1])


def _coverage(arr, r, fm, phis, A, tol=0.20, n_ang=13):
    """fraction of angles where the relative range error clears `tol`"""
    ths = np.linspace(-0.85, 0.85, n_ang)
    return np.mean([_crb_range(arr, t, r, fm, phis, A) / r <= tol for t in ths])


def exp7_usable_range(N=256, fc=40e9, beta=0.05, M=9, A0=30.0, tol=0.20):
    """A fixed decimated codebook over the whole sector - the user falls where it
    falls, no oracle centring. Reports where ranging still works at all."""
    arr = Array(N=N, fc=fc)
    print(f"\n[7] Usable ranging distance, fixed global codebook (no oracle centring)")
    print(f"    N={N} fc={fc/1e9:.0f}GHz  Rayleigh={2*(N*arr.d)**2/(C0/fc):.0f} m"
          f"  criterion: CRB_r/r <= {tol:.0%}")
    ranges = (6., 12., 18., 25., 35.)
    print(f"    {'scheme':<36} {'pilots':>7} | " + " ".join(f"r={r:<4.0f}" for r in ranges))
    for name, dec, b in (("narrowband, full DFT codebook", 1, None),
                         ("narrowband, every 8th beam", 8, None),
                         (f"wideband B/fc={beta}, every 8th beam", 8, beta),
                         (f"wideband B/fc={beta}, every 16th beam", 16, beta)):
        phis = arr.grid[::dec]
        fm = arr.fc if b is None else arr.subcarriers(b, M)
        A = A0 if b is None else A0 / np.sqrt(M)
        cov = " ".join(f"{_coverage(arr, r, fm, phis, A, tol):5.0%}" for r in ranges)
        print(f"    {name:<36} {len(phis):7d} | {cov}")
    print("    -> at a fixed pilot budget the frequency dimension roughly doubles the")
    print("       usable range and removes the angular blind spots entirely.")


def exp8_pilot_equivalence(N=256, fc=40e9, beta=0.05, M=9, A0=30.0, tol=0.20):
    """Per-pilot energy is fixed, so pilot count IS total energy - no energy trick."""
    arr = Array(N=N, fc=fc)
    print(f"\n[8] Pilot budget a wideband decimated codebook is worth")
    ranges = (6., 12., 18., 25.)
    print(f"    {'scheme':<36} {'pilots':>7} | " + " ".join(f"r={r:<4.0f}" for r in ranges))
    for name, dec, b in (("narrowband, full DFT codebook", 1, None),
                         ("narrowband, every 4th beam", 4, None),
                         ("narrowband, every 8th beam", 8, None),
                         (f"wideband B/fc={beta}, every 8th beam", 8, beta),
                         (f"wideband B/fc={beta}, every 4th beam", 4, beta)):
        phis = arr.grid[::dec]
        fm = arr.fc if b is None else arr.subcarriers(b, M)
        A = A0 if b is None else A0 / np.sqrt(M)
        cov = " ".join(f"{_coverage(arr, r, fm, phis, A, tol):5.0%}" for r in ranges)
        print(f"    {name:<36} {len(phis):7d} | {cov}")
    print("    -> 64 wideband pilots reproduce the full 256-pilot narrowband sweep;")
    print("       32 wideband pilots match 64 narrowband ones. Same PS-only front end.")
    print("    Design rule: pattern width in DFT beams is W ~ N^2 d (1-theta^2)/(2r),")
    print("    frequency dither is ~ beta N |theta| / 2. Narrowband needs a decimation")
    print("    factor below W; wideband needs it below W + dither. The dither does not")
    print("    create range information - W still has to exceed ~1 beam - it only lets a")
    print("    sparse grid sample a width it would otherwise step over.")


if __name__ == "__main__":
    arr = Array(N=512, fc=100e9)
    exp1_pattern_zooming(arr)
    exp2_dense_vs_sparse(arr)
    exp3_subcarrier_count(arr)
    exp4_vs_time_of_flight(arr)
    exp5_cost_of_dropping_ttd()
    exp6_gain_vs_fractional_bandwidth()
    exp7_usable_range()
    exp8_pilot_equivalence()
