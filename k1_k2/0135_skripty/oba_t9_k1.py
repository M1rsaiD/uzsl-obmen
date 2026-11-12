# к1 0135: «оба 99 %» = t9 в точке максимума смеси (−1,4: p = 0,19, s = 0,141)? Перекрёстная проверка к2 0152 (oba_t9_k2.py)
import numpy as np
from scipy import stats
rng=np.random.default_rng(135); n=10; tq=stats.t.ppf(0.995,n-1); N=20_000; B=2_000; p,s=0.19,0.141
z=rng.random((N,n))<p
d=np.where(z,-1.0,p/(1-p))+s*rng.standard_normal((N,n))
t9=(d.mean(1)-tq*d.std(1,ddof=1)/np.sqrt(n))>0
pc=np.empty(N,bool)
for i in range(0,N,500):                          # перцентильный бутстреп: нижняя граница 99 % = 0,5-й процентиль средних
    blk=d[i:i+500]; idx=rng.integers(0,n,(blk.shape[0],B,n))
    m=np.take_along_axis(blk[:,None,:].repeat(B,1),idx,2).mean(2)
    pc[i:i+500]=np.percentile(m,0.5,axis=1)>0
both=t9&pc
print(f"t9: {100*t9.mean():.2f} %  перцентильная: {100*pc.mean():.2f} %  оба: {100*both.mean():.2f} %")
print(f"t9 > 0, перцентильная — нет: {(t9&~pc).sum()}   перцентильная > 0, t9 — нет: {(pc&~t9).sum()}")
