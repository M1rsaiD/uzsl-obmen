import numpy as np
from scipy import stats
rng=np.random.default_rng(11)
tq=stats.t.ppf(0.995,9)
def run(gen,nsets=40000,nb=2000,n=10):
    fh=0; fl=0
    for k in range(0,nsets,400):
        d=gen((400,n)); m=d.mean(1); s=d.std(1,ddof=1)
        tlo=m-tq*s/np.sqrt(n)
        idx=rng.integers(0,n,(400,nb,n))
        bm=np.take_along_axis(np.repeat(d[:,None,:],nb,1),idx,2).mean(2)
        plo=np.quantile(bm,0.005,axis=1)
        H=(tlo>0)&(plo>0)
        G=stats.skew(d,axis=1,bias=False)
        fh+=H.sum(); fl+=(H&(G<-1)).sum()
    return fh/nsets, (fl/fh if fh else float('nan'))
gam=lambda k: (lambda sz: (k-rng.gamma(k,1,sz))/np.sqrt(k))   # skew = -2/sqrt(k)
for name,g in [('normal',lambda sz: rng.standard_normal(sz)),('skew -1 (k=4)',gam(4)),('skew -1.63 (k=1.5)',gam(1.5)),('skew -2 (k=1)',gam(1))]:
    fh,share=run(g); print(name,'false H',round(100*fh,2),'% ; share G1<-1 among false H',round(100*share,1),'%')
