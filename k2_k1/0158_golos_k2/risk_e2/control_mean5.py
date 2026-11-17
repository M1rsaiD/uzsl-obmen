"""Побочно к 22.5 (контрдовод «контроль = среднее 5 затравок M13»): при ρ пары ≈ 0 (rho_check.out) шум контроля σ²/5 вместо σ².
Та же нормальная модель и жадная цепочка, что risk_e2_mc.py; σ — из risk_e2_mc.json. Только ЦП."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent
R = json.load(open(HERE / "risk_e2_mc.json"))
THR, MINB, N, CH = 1.0081957667939179, 6, 200_000, 50_000
sig_b = np.array(R["inputs"]["sigma_base"])
c_b = np.array([0.443] * 5 + [0.399] * 5)            # база c (noise_summary.json, c_pooled val_micro_last)
acc = lambda d, a: ((d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)) & ((d[:, :5].mean(1) >= 0) if a else True)
for name, sig in (("база ×1", sig_b), ("c Sohn 0,25", sig_b * np.sqrt(0.25 / c_b))):
    for rule_a in (False, True):
        for ctrl in (1, 5):
            rng = np.random.default_rng(777 + ctrl + 10 * rule_a)
            first = np.zeros(14); pw = {1.0: 0, 2.0: 0}; a1 = 0
            for _ in range(N // CH):
                cur = rng.standard_normal((CH, 10)) * sig / np.sqrt(ctrl); anyacc = np.zeros(CH, bool)
                for mu in pw:
                    pw[mu] += acc(rng.standard_normal((CH, 10)) * sig + mu - cur, rule_a).sum()
                for j in range(1, 14):
                    e = rng.standard_normal((CH, 10)) * sig
                    ok = acc(e - cur, rule_a); a1 += ok.sum() if j == 1 else 0
                    anyacc |= ok; cur = np.where(ok[:, None], e, cur); first[j] += anyacc.sum()
            print(f"{name}, правило ({'а' if rule_a else 'б'}), контроль — {'среднее 5 затравок' if ctrl == 5 else '1 прогон (затравка 48)'}: "
                  f"α₁ {a1 / N:.3f}, α₅ {first[5] / N:.3f}, α₈ {first[8] / N:.3f}, α₁₃ {first[13] / N:.3f}; "
                  f"мощность поз. 1 μ=1/2: {pw[1.0] / N:.2f}/{pw[2.0] / N:.2f}")
