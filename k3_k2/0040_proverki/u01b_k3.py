"""к3 0040: независимая проверка U01b (ветка 120eced, src/mpk/sohn.py = блоб cd8a454) против определения поправки 2, 3.1.
sohn.py получен так: копия обмена k2_k3/0055_.../src/mpk/sohn.py (блоб 26f9f2b = база диффа) + git apply хунка sohn.py из
0154_u01b_045fd7d_120eced.diff → блоб cd8a454 (= ветка). torch — заглушка: проверяется только предобработка (numpy).
Своя формула — циклы в float64, кода к2 (hand_local) не использует. Синтетика, без данных участников."""
import itertools
import numpy as np
import mpk.sohn as s

rng = np.random.default_rng(20261116)
LH = [s.LOCAL[k] for k in s.LHAND]; RH = [s.LOCAL[k] for k in s.RHAND]

def mine(x, a):
    """Определение 3.1: p = (x, a·y); K_h = кадры t < 384 с точками 0 и 9; ℓ = медиана |p9 − p0| по K_h;
    u = (p_j − p_0)/ℓ; NaN, ±∞ → 0; обрезка [−10; 10]; столбцы: Л 0..20 (x, y), затем П."""
    x = x[:384].astype(np.float64)
    T = len(x); out = np.zeros((T, 84))
    for h, H in enumerate((LH, RH)):
        d = []
        for t in range(T):
            p0, p9 = x[t, H[0]] * [1, a], x[t, H[9]] * [1, a]
            if np.isfinite(p0).all() and np.isfinite(p9).all():
                d.append(np.hypot(*(p9 - p0)))
        if not d:
            continue                                  # нет ℓ → 42 нуля
        l = float(np.median(d))
        for t in range(T):
            for j in range(21):
                for c in range(2):
                    v = (x[t, H[j], c] - x[t, H[0], c]) * (a if c else 1) / l if l else np.inf
                    out[t, 42 * h + 2 * j + c] = 0.0 if not np.isfinite(v) else min(10.0, max(-10.0, v))
    return out

def record(T, kind):
    x = rng.uniform(0.2, 0.8, (T, 122, 2)).astype(np.float32)
    x[rng.random((T, 122)) < 0.05] = np.nan                      # пропуски точек
    if kind == "no_left":   x[:, LH] = np.nan                    # нет левой кисти
    if kind == "no09_right":                                     # у правой нет кадра, где есть и 0, и 9
        x[::2, RH[0]] = np.nan; x[1::2, RH[9]] = np.nan
    if kind == "palm0":    x[:, LH[9]] = x[:, LH[0]]             # ладонь 0 → ±∞ → 0
    if kind == "late_only":                                      # правая кисть есть только после кадра 384
        x[:384, RH] = np.nan
    if kind == "far":     x[:, LH[8]] += 5.0                      # выход за ±10 ладоней → обрезка
    return x

res = []
for T, kind in itertools.product((1, 2, 3, 40, 384, 611), ("plain", "no_left", "no09_right", "palm0", "late_only", "far")):
    if kind == "late_only" and T <= 384: continue
    x = record(T, kind); a = float(rng.choice([16 / 9, 9 / 16, 4 / 3, 1.0]))
    y = s.preprocess_u(x, a, ("u01b",))
    y4 = s.preprocess_u(x, a, ("u04",)); y41 = s.preprocess_u(x, a, ("u04", "u01b"))
    base = s.preprocess(x[:, :s.NUM_NODES])
    ref = mine(x, a)
    ok708 = y.shape[1] == 792 and np.array_equal(y[:, :708], base, equal_nan=False)
    ok_u04 = y41.shape[1] == 1242 and np.array_equal(y41[:, :1158], y4) and np.array_equal(y41[:, 1158:], y[:, 708:])
    err = np.abs(y[:, 708:] - ref).max() / max(1.0, np.abs(ref).max())
    zero4 = not y[:, 708 + np.array([0, 1, 42, 43])].any()
    res.append((T, kind, ok708, ok_u04, err, zero4))
    print(f"T={T:4d} {kind:11s} 708=Sohn:{ok708} u04+u01b:{ok_u04} max rel err={err:.2e} 4 нуля:{zero4} "
          f"Л≠0:{bool(y[:, 708:750].any())} П≠0:{bool(y[:, 750:].any())}")
assert all(r[2] and r[3] and r[4] < 1e-6 and r[5] for r in res), "расхождение"
print("ВСЁ СОВПАЛО:", len(res), "записей")
