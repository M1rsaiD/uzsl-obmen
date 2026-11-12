# к1 0133: нижний край у смеси при заданной асимметрии (малое p, большой s) — функции из 0132
import numpy as np
exec(open('k1_k2/0132_skripty/max_pri_asimm.py',encoding='utf-8').read().split('assert')[0])
for g in (-0.7,-1.4):
    for p in (0.01,0.02,0.05,0.10):
        s=s_for(p,g); print(f"асимметрия {g}: p = {p:.2f}, s = {s:.3f}, ложное H = {rate(p,s):.2f} %")
