import numpy as np
from scipy import stats
rng=np.random.default_rng(23); n=10; tq=stats.t.ppf(0.995,n-1)
def t9(gen,N=2_000_000,chunk=200_000):
    fh=fw=gh=0
    for _ in range(N//chunk):
        d=gen((chunk,n)); m=d.mean(1); s=d.std(1,ddof=1); h=tq*s/np.sqrt(n)
        H=(m-h)>0; Wo=(m+h)<0; G=stats.skew(d,axis=1,bias=False)
        fh+=H.sum(); fw+=Wo.sum(); gh+=(H&(G<-1)).sum()
    return 100*fh/N, 100*fw/N, 100*gh/max(fh,1)
gam=lambda k,sgn=1: (lambda sz: sgn*(k-rng.gamma(k,1,sz))/np.sqrt(k))
tdist=lambda df: (lambda sz: rng.standard_t(df,sz))
for name,g in [('normal',lambda sz: rng.standard_normal(sz)),('t5',tdist(5)),('t3',tdist(3)),('skew -1',gam(4)),('skew -2',gam(1)),('skew +1.63',gam(1.5,-1))]:
    fh,fw,sh=t9(g); print(f'{name:10s} falseH {fh:.3f}%  false worse {fw:.3f}%  G1<-1 among falseH {sh:.1f}%')
print('SE at 40000, p=4.57%:', round(100*np.sqrt(0.0457*0.9543/40000),3),'pp')
