# к1 0158: G2 (к2 0178) — может ли `verdict` (lo > 0) сказать «лучше» при точной нижней границе 0 на разных моделях.
# boot() turnir_eval.py 964ecba, строки 500–510, — дословно (ast); рядом — та же квантиль в точной арифметике (Fraction).
# Случаи — 12 исполнителей (как LOSO корпуса), разные знаменатели, разности по исполнителям — малые целые со знаком.
import ast, sys
from fractions import Fraction as F
import numpy as np
src = open(sys.argv[1], encoding="utf-8").read()
i = src.index("\ndef boot("); j = src.index("\n\n\ndef ", i)   # в master к2 строки 42–43 замаскированы — берём текст функции
fn = ast.parse(src[i:j]).body[0]; assert fn.name == "boot"
ns = {"np": np, "BOOT": {"UzSL": (2000, 7)}}; exec(compile(ast.Module([fn], []), "turnir_eval", "exec"), ns)
rng, hits, zero, tried = np.random.default_rng(1), 0, 0, 0
for t in range(6000):
    if t < 3000:   # (1) разные знаменатели, разности — малые целые со знаком
        den = rng.integers(60, 240, 12).astype(float); num = rng.choice([-2, -1, 0, 0, 0, 1, 1, 2, 3], 12).astype(float)
    else:          # (2) худший для float случай: знаменатели из трёх значений, разности почти все 0 и ±1
        den = rng.choice([90., 120., 210.], 12); num = rng.choice([-1, 0, 0, 0, 0, 0, 1, 1], 12).astype(float)
    _, lo, _ = ns["boot"](num, den, 0.95, "UzSL", True)
    # точная квантиль: те же индексы бутстрепа, среднее в дробях, та же линейная интерполяция np.quantile
    b = np.random.default_rng(7).integers(0, 12, (2000, 12))
    ex = sorted(sum(F(int(num[j]), int(den[j])) for j in row) / 12 for row in b)
    h = F(2000 - 1) * F(1, 40); k = int(h); lo_ex = ex[k] + (h - k) * (ex[k + 1] - ex[k])
    tried += 1
    if lo_ex == 0:
        zero += 1
        if lo != 0:
            hits += 1
            if hits <= 3:
                print(f"случай {t}: точная нижняя граница 0, float {lo:+.3e} → verdict = «{'лучше' if lo > 0 else 'хуже/не уст.'}»")
print(f"случаев {tried}; точная граница = 0 — {zero}; из них float ≠ 0 — {hits}")
