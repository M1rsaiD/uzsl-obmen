# к1 0132: максимум ложного «H» у смеси при ЗАДАННОЙ асимметрии (s подбирается под асимметрию), t9 99 %, n = 10
import numpy as np
from scipy import stats
rng=np.random.default_rng(132); n=10; tq=stats.t.ppf(0.995,n-1)
def mom(p):
    a,b=-1.0,p/(1-p); return p*a*a+(1-p)*b*b, p*a**3+(1-p)*b**3
def s_for(p,g):                      # s, при котором асимметрия смеси = g (g < 0); None, если недостижимо
    m2,m3=mom(p); need=(m3/g)**(2/3)
    return None if need<m2 else np.sqrt(need-m2)
def skew(p,s): m2,m3=mom(p); return m3/(m2+s*s)**1.5
def rate(p,s,N=1_000_000,chunk=200_000):
    h=0
    for _ in range(N//chunk):
        z=rng.random((chunk,n))<p
        d=np.where(z,-1.0,p/(1-p))+s*rng.standard_normal((chunk,n))
        h+=((d.mean(1)-tq*d.std(1,ddof=1)/np.sqrt(n))>0).sum()
    return 100*h/N
assert abs(skew(0.2,s_for(0.2,-1.3))+1.3)<1e-9
for g in (-1.4,-1.58):
    print(f"асимметрия {g}:  p     s      s/выигрыш  ложное H %  (1-p)^10 %")
    for p in np.arange(0.12,0.235,0.01):
        s=s_for(p,g)
        if s is None: continue
        print(f"                {p:.2f}  {s:.3f}  {s/(p/(1-p)):6.2f}    {rate(p,s):7.2f}     {100*(1-p)**n:6.2f}")
