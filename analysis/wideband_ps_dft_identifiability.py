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


if __name__ == "__main__":
    arr = Array(N=512, fc=100e9)
    exp1_pattern_zooming(arr)
    exp2_dense_vs_sparse(arr)
    exp3_subcarrier_count(arr)
    exp4_vs_time_of_flight(arr)
