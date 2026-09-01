import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-DFT-Beam/d85f13c3-2355-5b36-b224-9a8f4918618e/scratchpad')
import numpy as np
from wang import (SimulationConfig, PAPER_CFG, element_offsets, far_field_steering,
                  dft_codebook, main_lobe_width, candidate_angle_indices, C0)

cfg = PAPER_CFG
u_grid, W = dft_codebook(cfg)

def steering_at(cfg, u, r, f):
    """their near_field_steering, but at an arbitrary frequency (PS array: spacing fixed at f_c)"""
    y = element_offsets(cfg)
    rn = np.sqrt(r**2 + y**2 - 2.0*r*u*y)
    return np.exp(-1j*2*np.pi*(rn - r)*f/C0)/np.sqrt(cfg.N)

def coarse_r(z, u_true):
    """their coarse range estimator, verbatim formula"""
    idx = candidate_angle_indices(z, u_grid, k=3)
    c = int(np.rint(np.median(idx)))
    width, _ = main_lobe_width(z, u_grid, c, threshold=0.5)
    return cfg.N*cfg.spacing*(1.0 - u_grid[c]**2)/max(width,1e-12), u_grid[c]

print("Wang 2026's coarse range estimator, run with THEIR code, no noise.")
print(f"Config: their PAPER_CFG  N={cfg.N}, f_c={cfg.fc/1e9:.0f} GHz, r in [{cfg.r_min},{cfg.r_max}] m\n")
print(f"{'u':>5} {'r true':>7} | {'narrowband':>11} {'err':>7} | "
      + "  ".join(f"B/fc={b:<4.2f}" for b in (0.05,0.10,0.20)))
for u,r in ((0.0,10.),(0.3,10.),(0.3,30.),(0.6,10.),(0.6,30.),(0.6,60.)):
    zn = np.abs(W @ np.conjugate(steering_at(cfg,u,r,cfg.fc)))
    rn,_ = coarse_r(zn,u)
    cells=[]
    for beta in (0.05,0.10,0.20):
        M=65; fm = cfg.fc + beta*cfg.fc/M*(np.arange(1,M+1)-1-(M-1)/2)
        P = np.mean([np.abs(W @ np.conjugate(steering_at(cfg,u,r,f)))**2 for f in fm], axis=0)
        rw,_ = coarse_r(np.sqrt(P),u)
        cells.append(f"{rw:7.2f} ({100*(rw-r)/r:+5.1f}%)")
    print(f"{u:5.1f} {r:7.1f} | {rn:11.2f} {100*(rn-r)/r:+6.1f}% | " + "  ".join(cells))
