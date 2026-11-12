# к1 0132: максимум по p при асимметрии −0,7 / −1,0 / −1,2 (внутри диапазона сноски) — как max_pri_asimm.py
import numpy as np
src=open('max_pri_asimm.py',encoding='utf-8').read().split('assert')[0]
exec(src)
for g in (-0.7,-1.0,-1.2):
    best=max(((rate(p,s_for(p,g),N=600_000),p,s_for(p,g)) for p in np.arange(0.12,0.40,0.01) if s_for(p,g) is not None))
    print(f"асимметрия {g}: максимум {best[0]:.2f} % при p = {best[1]:.2f}, s = {best[2]:.3f}")
