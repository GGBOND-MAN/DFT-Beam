import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-DFT-Beam/d85f13c3-2355-5b36-b224-9a8f4918618e/scratchpad')
import numpy as np
from wang import PAPER_CFG, element_offsets, dft_codebook, main_lobe_width, candidate_angle_indices, C0
cfg=PAPER_CFG; u_grid,W=dft_codebook(cfg); Y=element_offsets(cfg)
def pat(u,r,f):
    rn=np.sqrt(r**2+Y**2-2*r*u*Y)
    return np.abs(W@np.conjugate(np.exp(-1j*2*np.pi*(rn-r)*f/C0)/np.sqrt(cfg.N)))
def obs(u,r,beta,M=65):
    if beta==0: z=pat(u,r,cfg.fc)
    else:
        fm=cfg.fc+beta*cfg.fc/M*(np.arange(1,M+1)-1-(M-1)/2)
        z=np.sqrt(np.mean([pat(u,r,f)**2 for f in fm],axis=0))
    i=candidate_angle_indices(z,u_grid,k=3); c=int(np.rint(np.median(i)))
    w,_=main_lobe_width(z,u_grid,c,threshold=0.5); return w,u_grid[c]

TRAIN=[(u,r) for u in (0.20,0.40,0.60) for r in (10.,18.,30.)]
TEST =[(u,r) for u in (0.15,0.30,0.45,0.55,0.70) for r in (8.,14.,22.,40.)]
print("Target isolated: recover the NARROWBAND width W_c from the wideband W_meas.")
print("Wang's own narrowband error is then untouched - it is not what this claims to fix.")
print("Law:  W_c_hat = (W_meas - beta*u) / B(beta),  B calibrated on 9 points.\n")
print(f"{'beta':>5} | {'B':>6} | {'TRAIN  raw':>11} {'corrected':>10} | {'TEST  raw':>10} {'corrected':>10}")
for beta in (0.05,0.10,0.20,0.30):
    D={p:(obs(p[0],p[1],0.)[0], *obs(p[0],p[1],beta)) for p in TRAIN+TEST}   # Wc, Wm, uhat
    num=den=0.
    for p in TRAIN:
        Wc,Wm,uh=D[p]; x=Wm-beta*abs(uh); num+=x*Wc; den+=Wc*Wc
    B=num/den
    def err(S,corr):
        e=[]
        for p in S:
            Wc,Wm,uh=D[p]
            wh=(Wm-beta*abs(uh))/B if corr else Wm
            e.append(abs(wh-Wc)/Wc)
        return 100*np.mean(e)
    print(f"{beta:5.2f} | {B:6.3f} | {err(TRAIN,0):10.1f}% {err(TRAIN,1):9.1f}% |"
          f" {err(TEST,0):9.1f}% {err(TEST,1):9.1f}%")
