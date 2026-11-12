import numpy as np
from scipy import stats
rng=np.random.default_rng(7)
# (a) circle in pixels, frame W:H; affine in frame fractions: rotation ±30°, shear ±0.15 (x += sh*y); ratio of per-axis std
def iqr_med(WH, n=20000):
    t=np.linspace(0,2*np.pi,64,endpoint=False)
    X=np.cos(t); Y=np.sin(t)              # pixels, isotropic circle
    W,H=WH
    x=X/W; y=Y/H                           # frame fractions
    rho0=np.std(y)/np.std(x)
    th=np.deg2rad(rng.uniform(-30,30,n)); sh=rng.uniform(-0.15,0.15,n)
    c,s=np.cos(th),np.sin(th)
    xr=c[:,None]*x - s[:,None]*y; yr=s[:,None]*x + c[:,None]*y
    xr=xr+sh[:,None]*yr
    r=(yr.std(1)/xr.std(1))/rho0
    q1,m,q3=np.quantile(r,[.25,.5,.75])
    return round((q3-q1)/m,3)
print('(a) circle IQR/median rho\'/rho:', {k:iqr_med(v) for k,v in {'16:9':(16,9),'4:3':(4,3),'1:1':(1,1),'9:16':(9,16)}.items()})
# (b) false 'H established' under skewed null: d = (3 - chi2_3)/sqrt(6), mean 0
def both_rate(gen, nsets=20000, nb=2000, n=10):
    hits=0; tq=stats.t.ppf(0.995,n-1)
    for k in range(0,nsets,500):
        d=gen((500,n))
        m=d.mean(1); s=d.std(1,ddof=1)
        tlo=m-tq*s/np.sqrt(n)
        idx=rng.integers(0,n,(500,nb,n))
        bm=np.take_along_axis(np.repeat(d[:,None,:],nb,1),idx,2).mean(2)
        plo=np.quantile(bm,0.005,axis=1)
        hits+=np.sum((tlo>0)&(plo>0))
    return hits/nsets
neg=lambda sz:(3-rng.chisquare(3,sz))/np.sqrt(6)
print('(b) false H (both rule), 3-chi2_3:', both_rate(neg), ' normal:', both_rate(lambda sz: rng.standard_normal(sz)))
# (c) G1 (bias=False) flags g<-1 at n=10
def flag(gen,N=200000):
    d=gen((N,10)); g=stats.skew(d,axis=1,bias=False); gb=stats.skew(d,axis=1,bias=True)
    return round(np.mean(g<-1),3), round(np.mean(gb<-1),3), round(g.mean(),2)
print('(c) 3-chi2_3 [G1<-1, g1<-1, mean G1]:', flag(neg), ' normal:', flag(lambda sz: rng.standard_normal(sz)))
print('sqrt(n(n-1))/(n-2) n=10:', round(np.sqrt(90)/8,3))
print('(d) share control pretraining:', round(2.9/105.1*100,2), round(2.9/79.0*100,2))
print('(e) 1/sqrt(1.16..1.21):', round(1/np.sqrt(1.16),3), round(1/np.sqrt(1.21),3), ' t9 0.995:', round(stats.t.ppf(0.995,9),3))
