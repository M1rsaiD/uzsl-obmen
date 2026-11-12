# к1 0131: при каком разбросе "остальных" смесь даёт ~11 % ложного «H» (t9 99 %, n = 10)
import numpy as np
from scipy import stats
rng=np.random.default_rng(131); n=10; tq=stats.t.ppf(0.995,n-1)
def skew_exact(p,s):
    a,b=-1.0,p/(1-p)
    m2=p*a*a+(1-p)*b*b+s*s; m3=p*a**3+(1-p)*b**3   # шум нормальный: в m3 не входит
    return m3/m2**1.5
def rate(p,s,N=1_000_000,chunk=200_000):
    h=0
    for _ in range(N//chunk):
        z=rng.random((chunk,n))<p
        d=np.where(z,-1.0,p/(1-p))+s*rng.standard_normal((chunk,n))
        h+=((d.mean(1)-tq*d.std(1,ddof=1)/np.sqrt(n))>0).sum()
    return 100*h/N
assert abs(skew_exact(0.2,0.1)+1.414)<1e-3 and abs(skew_exact(0.1,0.3)+1.095)<1e-3
print("p     s     s/mean  skew   false_H%   (1-p)^10 %")
for p,s in [(0.2,0.05),(0.2,0.1),(0.2,0.15),(0.2,0.2),(0.2,0.25),(0.2,0.3),
            (0.15,0.22),(0.25,0.1),(0.3,0.1),(0.3,0.3)]:
    print(f"{p:<5} {s:<5} {s/(p/(1-p)):6.2f} {skew_exact(p,s):6.2f} {rate(p,s):8.2f}   {100*(1-p)**n:6.2f}")
