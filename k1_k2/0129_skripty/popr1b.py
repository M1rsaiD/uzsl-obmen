import numpy as np
from scipy import stats
rng=np.random.default_rng(31); n=10; tq=stats.t.ppf(0.995,n-1)
def t9(gen,N=2_000_000,chunk=200_000):
    fh=gh=0
    for _ in range(N//chunk):
        d=gen((chunk,n)); m=d.mean(1); s=d.std(1,ddof=1)
        H=(m-tq*s/np.sqrt(n))>0; G=stats.skew(d,axis=1,bias=False)
        fh+=H.sum(); gh+=(H&(G<-1)).sum()
    return 100*fh/N, 100*gh/max(fh,1)
def bim(s): return lambda sz: rng.choice([-1.0,1.0],sz)+s*rng.standard_normal(sz)
def mix(p,s):
    def g(sz):
        z=rng.random(sz)<p
        return np.where(z,-1.0,p/(1-p))+s*rng.standard_normal(sz)
    return g
forms=[('uniform',lambda sz: rng.uniform(-1,1,sz)),('bimodal s=0.3',bim(0.3)),('bimodal s=0.1',bim(0.1)),
       ('mix p=.2 s=.3',mix(.2,.3)),('mix p=.2 s=.1',mix(.2,.1)),('mix p=.1 s=.3',mix(.1,.3)),('mix p=.3 s=.3',mix(.3,.3))]
for name,g in forms:
    sk=stats.skew(g((2_000_000,)))
    fh,sh=t9(g); print(f'{name:14s} skew {sk:+.2f}  falseH {fh:.3f}%  G1<-1 among falseH {sh:.1f}%')
