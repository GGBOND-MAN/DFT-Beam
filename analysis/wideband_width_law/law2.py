import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-DFT-Beam/d85f13c3-2355-5b36-b224-9a8f4918618e/scratchpad')
import numpy as np
from scipy.optimize import minimize_scalar
from wang import PAPER_CFG, element_offsets, dft_codebook, main_lobe_width, candidate_angle_indices, C0
cfg=PAPER_CFG; u_grid,W=dft_codebook(cfg); Y=element_offsets(cfg)

def pat(u,r,f):
    rn=np.sqrt(r**2+Y**2-2*r*u*Y)
    return np.abs(W@np.conjugate(np.exp(-1j*2*np.pi*(rn-r)*f/C0)/np.sqrt(cfg.N)))
def observe(u,r,beta,M=65):
    if beta==0: z=pat(u,r,cfg.fc)
    else:
        fm=cfg.fc+beta*cfg.fc/M*(np.arange(1,M+1)-1-(M-1)/2)
        z=np.sqrt(np.mean([pat(u,r,f)**2 for f in fm],axis=0))
    idx=candidate_angle_indices(z,u_grid,k=3); c=int(np.rint(np.median(idx)))
    w,_=main_lobe_width(z,u_grid,c,threshold=0.5)
    return w, u_grid[c]

def r_of(w,u):  return cfg.N*cfg.spacing*(1-u**2)/max(w,1e-12)

TRAIN=[(u,r) for u in (0.20,0.40,0.60) for r in (10.,18.,30.)]
TEST =[(u,r) for u in (0.15,0.30,0.45,0.55,0.70) for r in (8.,14.,22.,40.)]

print("O(1) corrected law:   r = N d (1-u^2) / [ (W_meas - beta*u) / B(beta) ]")
print("A fixed to beta (theory: full in-band centre travel); only B(beta) calibrated,")
print("on 9 training points, evaluated on 20 held-out points.\n")
print(f"{'beta':>5} | {'B cal':>7} | {'TRAIN uncorr':>13} {'corr':>8} | {'TEST uncorr':>12} {'corr':>8}")
Bs={}
for beta in (0.05,0.10,0.20,0.30):
    obs={p:observe(p[0],p[1],beta) for p in TRAIN+TEST}
    def cost(B):
        e=[]
        for (u,r) in TRAIN:
            w,uh=obs[(u,r)]
            e.append((r_of(max((w-beta*abs(uh))/B,1e-9),uh)-r)/r)
        return float(np.mean(np.square(e)))
    B=minimize_scalar(cost,bounds=(0.2,1.5),method='bounded').x
    Bs[beta]=B
    def errs(S,corrected):
        out=[]
        for (u,r) in S:
            w,uh=obs[(u,r)]
            ww=(w-beta*abs(uh))/B if corrected else w
            out.append(abs(r_of(max(ww,1e-9),uh)-r)/r)
        return 100*np.mean(out)
    print(f"{beta:5.2f} | {B:7.4f} | {errs(TRAIN,0):12.1f}% {errs(TRAIN,1):7.1f}% |"
          f" {errs(TEST,0):11.1f}% {errs(TEST,1):7.1f}%")
print("\nB(beta) calibrated:", {k:round(v,4) for k,v in Bs.items()})
b=np.array(list(Bs)); B=np.array(list(Bs.values()))
c=np.polyfit(b,1/B,1); print(f"1/B fits  {c[0]:.3f}*beta + {c[1]:.3f}   (residual {np.abs(np.polyval(c,b)-1/B).max():.4f})")
