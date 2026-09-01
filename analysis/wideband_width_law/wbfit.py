import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-DFT-Beam/d85f13c3-2355-5b36-b224-9a8f4918618e/scratchpad')
import numpy as np
from wang import (PAPER_CFG, element_offsets, dft_codebook, main_lobe_width,
                  candidate_angle_indices, C0)
cfg=PAPER_CFG; u_grid,W=dft_codebook(cfg)
Y=element_offsets(cfg)

def pat(u,r,f):
    rn=np.sqrt(r**2+Y**2-2*r*u*Y)
    return np.abs(W@np.conjugate(np.exp(-1j*2*np.pi*(rn-r)*f/C0)/np.sqrt(cfg.N)))

def measure(u,r,beta,M=9):
    if beta==0: return pat(u,r,cfg.fc)
    fm=cfg.fc+beta*cfg.fc/M*(np.arange(1,M+1)-1-(M-1)/2)
    return np.sqrt(np.mean([pat(u,r,f)**2 for f in fm],axis=0))

def wang_est(z):
    idx=candidate_angle_indices(z,u_grid,k=3); c=int(np.rint(np.median(idx)))
    w,_=main_lobe_width(z,u_grid,c,threshold=0.5)
    return cfg.N*cfg.spacing*(1-u_grid[c]**2)/max(w,1e-12), u_grid[c]

def wb_fit(z,beta,M=9,u0=None):
    """fit the exact wideband summed pattern instead of using the narrowband width law"""
    us=np.linspace(max(u0-0.05,-0.95),min(u0+0.05,0.95),41) if u0 is not None else np.linspace(-0.9,0.9,181)
    inv=1/np.linspace(cfg.r_min,cfg.r_max,300)
    best=(np.inf,None)
    for uu in us:
        for iv in inv:
            g=measure(uu,1/iv,beta,M)
            v=-( (z*g).sum()**2 )/max((g*g).sum(),1e-30)
            if v<best[0]: best=(v,(uu,1/iv))
    return best[1]

print("Replacing the narrowband width law with a fit to the exact wideband pattern.")
print("Wang 2026 PAPER_CFG (N=512, f_c=100 GHz), M=9 subcarriers, no noise.\n")
print(f"{'u':>5} {'r':>6} {'beta':>6} | {'Wang width law':>15} {'err':>8} | {'wideband fit':>13} {'err':>8}")
a=b=n=0
for u,r in ((0.3,10.),(0.45,10.),(0.6,10.),(0.6,20.),(0.75,10.)):
    for beta in (0.05,0.10,0.20):
        z=measure(u,r,beta)
        r0,uh=wang_est(z)
        uf,rf=wb_fit(z,beta,9,uh)
        e0,e1=100*(r0-r)/r,100*(rf-r)/r
        a+=abs(e0); b+=abs(e1); n+=1
        print(f"{u:5.2f} {r:6.1f} {beta:6.2f} | {r0:15.2f} {e0:+7.1f}% | {rf:13.2f} {e1:+7.1f}%")
print(f"\nmean |error|:  Wang width law {a/n:.1f}%   wideband fit {b/n:.1f}%")
