# Аудитор: сверка вывода к3 0045 разд. 1 (√0,2; границы 3.1; потолок при s = 16) — ЦП, синтетика
import math, numpy as np
rng = np.random.default_rng(0)
d, n, p_keep = 384, 4000, 0.2
out = {}
for c_target in (0.3, 0.6, 0.9):
    f = rng.standard_normal(d) * 1.5; f = f / (1 + np.exp(-f))            # swish, как признак Sohn
    u = rng.standard_normal(d); u -= u @ f / (f @ f) * f; u /= np.linalg.norm(u)
    fh = f / np.linalg.norm(f)
    w = c_target * fh + math.sqrt(1 - c_target**2) * u                  # прототип с cos(w, f) = c_target
    m = rng.random((n, d)) < p_keep
    fd = m * f / p_keep                                                  # dropout до нормировки (сейчас)
    cos_now = (fd @ w) / (np.linalg.norm(fd, axis=1) * np.linalg.norm(w))
    fdB = m * fh / p_keep                                                # (Б): нормировка до dropout
    logB = fdB @ (w / np.linalg.norm(w))
    out[c_target] = (round(cos_now.mean() / c_target, 3), round(logB.mean() / c_target, 3))
print("ratio train/eval: сейчас, (Б):", out, " sqrt(0.2) =", round(math.sqrt(0.2), 4))
for name, C in (("Slovo", 1001), ("S+B", 1034), ("Bukva", 33)):
    b = math.log(1 + 9 * C)
    ceil = math.exp(16 * math.sqrt(0.2)) / (math.exp(16 * math.sqrt(0.2)) + C - 1)
    print(f"{name}: ln(1+9C) = {b:.3f}; s(A) = {b / math.sqrt(0.2):.2f}; потолок p при s=16 сейчас = {ceil:.3f}")
