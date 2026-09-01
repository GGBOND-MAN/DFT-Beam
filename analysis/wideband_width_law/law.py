import sys; sys.path.insert(0,'/tmp/claude-0/-home-user-DFT-Beam/d85f13c3-2355-5b36-b224-9a8f4918618e/scratchpad')
import numpy as np
from wang import PAPER_CFG, element_offsets, dft_codebook, main_lobe_width, candidate_angle_indices, C0
cfg=PAPER_CFG; u_grid,W=dft_codebook(cfg); Y=element_offsets(cfg)

def pat(u,r,f):
    rn=np.sqrt(r**2+Y**2-2*r*u*Y)
    return np.abs(W@np.conjugate(np.exp(-1j*2*np.pi*(rn-r)*f/C0)/np.sqrt(cfg.N)))
def meas_width(u,r,beta,M=65):
    if beta==0: z=pat(u,r,cfg.fc)
    else:
        fm=cfg.fc+beta*cfg.fc/M*(np.arange(1,M+1)-1-(M-1)/2)
        z=np.sqrt(np.mean([pat(u,r,f)**2 for f in fm],axis=0))
    idx=candidate_angle_indices(z,u_grid,k=3); c=int(np.rint(np.median(idx)))
    w,_=main_lobe_width(z,u_grid,c,threshold=0.5)
    return w

print("Testing the structural claim  W_meas = A(beta)*u + B(beta)*W_c")
print("W_c taken as the MEASURED narrowband width (beta=0), so any error in the")
print("narrowband law itself cancels out and only the wideband effect is tested.\n")
us=np.array([0.15,0.30,0.45,0.60,0.75]); rs=np.array([8.,12.,20.,30.,45.])
print(f"{'beta':>5} | {'A fitted':>9} {'A theory':>9} | {'B fitted':>9} {'B theory':>9} | {'R^2':>7} {'n':>4}")
for beta in (0.05,0.10,0.20,0.30):
    X=[];y=[]
    for u in us:
        for r in rs:
            Wc=meas_width(u,r,0.0); Wm=meas_width(u,r,beta)
            X.append([u,Wc]); y.append(Wm)
    X=np.array(X); y=np.array(y)
    coef,res,*_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@coef; ss=1-((y-pred)**2).sum()/((y-y.mean())**2).sum()
    sp,sm=np.sqrt(1+beta/2),np.sqrt(1-beta/2); G=(1-beta**2/4)**0.25
    print(f"{beta:5.2f} | {coef[0]:9.4f} {G*(sp-sm):9.4f} | {coef[1]:9.4f} {G*(sp+sm)/2:9.4f} |"
          f" {ss:7.4f} {len(y):4d}")
