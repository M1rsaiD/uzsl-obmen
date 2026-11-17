# к1 0138: независимая проверка hand_local (U01b, ветка 120eced, блоб sohn.py cd8a454) — своя формула (float64, по кадрам)
# и граничные случаи. Запуск: PYTHONPATH=stub:. python3 -I u01b_k1.py  (stub — заглушка torch только для импорта)
import sys; sys.path[:0] = ['stub', '.']
import numpy as np
import mpk.sohn as s
rng = np.random.default_rng(138)
L, R = s.LH_LOC, s.RH_LOC

def mine(x, aspect, max_len=384):
    """Определение поправки 2, 3.1, буквально: p = (x, a·y); K_h — кадры t < T′ с точками 0 и 9; ℓ_h — медиана; u = (p − p0)/ℓ_h."""
    xp = np.asarray(x[:max_len, :118], np.float64) * [1.0, aspect]
    cols = []
    for H in (L, R):
        q = xp[:, H]                                    # [T′, 21, 2]
        K = [t for t in range(len(q)) if np.all(np.isfinite(q[t, 0])) and np.all(np.isfinite(q[t, 9]))]
        ell = np.median([np.hypot(*(q[t, 9] - q[t, 0])) for t in K]) if K else None
        blk = np.zeros((len(q), 42))
        if ell is not None:
            for t in range(len(q)):
                for j in range(21):
                    for c in range(2):
                        v = (q[t, j, c] - q[t, 0, c]) / ell if ell != 0 else np.nan
                        blk[t, 2 * j + c] = 0.0 if not np.isfinite(v) else min(10.0, max(-10.0, v))
        cols.append(blk)
    return np.concatenate(cols, 1)

def rec(T, miss=0.1):
    x = rng.random((T, 122, 2)).astype(np.float32)
    x[rng.random((T, 122)) < miss] = np.nan
    return x

def check(name, x, a, cond=None):
    got = s.preprocess_u(x, a, ("u01b",))
    want = mine(x, a)
    assert got.shape[1] == 792, got.shape
    base = s.preprocess(x[:, :118])
    assert np.array_equal(got[:, :708], base), name + ": первые 708 ≠ Sohn"
    err = np.max(np.abs(got[:, 708:] - want)) if len(want) else 0.0
    assert err < 5e-6, (name, err)
    both = s.preprocess_u(x, a, ("u04", "u01b"))
    assert np.array_equal(both[:, -84:], got[:, 708:]) and both.shape[1] == 708 + 450 + 84, name + ": порядок u04 → u01b"
    assert np.array_equal(both[:, :1158], s.preprocess_u(x, a, ("u04",))), name + ": u04 перед u01b не тот"
    assert not got[:, 708:][:, [0, 1, 42, 43]].any(), name + ": нулевые столбцы"
    if cond: cond(got[:, 708:])
    print(f"{name:<44} T={len(x):>4} a={a:.3f}  max|Δ|={err:.1e}  ok")

# 1) обычные записи, разные H/W
for a in (16 / 9, 9 / 16, 1.0):
    check(f"случайная, пропуски 10 %", rec(120), a)
# 2) T > 384: ладонь только по первым 384 кадрам (у кадров ≥ 384 ладонь ×100)
x = rec(500, 0.0); x[384:, R, :] *= 100; x[384:, L, :] *= 100
check("T = 500, хвост с огромной ладонью", x, 16 / 9, lambda f: None)
# 3) запястье пропущено в части кадров → в этих кадрах 42 нуля кисти
x = rec(60, 0.0); x[10:20, L[0]] = np.nan
check("нет запястья Л в кадрах 10–19", x, 1.0, lambda f: (_ for _ in ()).throw(AssertionError) if f[10:20, :42].any() else None)
# 4) точки 9 нет нигде → ℓ_h нет → 42 нуля во всех кадрах
x = rec(60, 0.0); x[:, R[9]] = np.nan
check("нет точки 9 П во всех кадрах", x, 1.0, lambda f: (_ for _ in ()).throw(AssertionError) if f[:, 42:].any() else None)
# 5) ладонь 0 (9 совпадает с 0) → деление на 0 → 0
x = rec(40, 0.0); x[:, L[9]] = x[:, L[0]]
check("ладонь Л = 0", x, 1.0, lambda f: (_ for _ in ()).throw(AssertionError) if f[:, :42].any() else None)
# 6) ±∞ в координатах точки
x = rec(40, 0.0); x[5, L[3], 0] = np.inf; x[6, R[4], 1] = -np.inf
check("±∞ у точек 3 Л и 4 П", x, 1.0)
# 7) обрезка: точка далеко → ±10
x = rec(40, 0.0); x[:, R[8]] = x[:, R[0]] + 50.0
check("точка 8 П далеко → обрезка", x, 1.0, lambda f: (_ for _ in ()).throw(AssertionError) if not np.all(np.abs(f[:, 42 + 16:42 + 18]) == 10) else None)
# 8) вся кисть пропущена
x = rec(40, 0.0); x[:, L] = np.nan
check("кисти Л нет совсем", x, 9 / 16, lambda f: (_ for _ in ()).throw(AssertionError) if f[:, :42].any() else None)

# 9) отражение: x → 1 − x и обмен кистей ⇒ блок Л отражённой = блок П исходной с x → −x (ладонь не меняется)
x = rec(80, 0.05); a = 16 / 9
xf = x.copy(); xf[..., 0] = 1 - xf[..., 0]; xf[:, L], xf[:, R] = x[:, R].copy(), x[:, L].copy(); xf[:, L, 0] = 1 - x[:, R, 0]; xf[:, R, 0] = 1 - x[:, L, 0]
f, g = s.hand_local(x[:384, :118] * np.array([1, a], np.float32)), s.hand_local(xf[:384, :118] * np.array([1, a], np.float32))
sgn = np.tile([-1.0, 1.0], 21)
assert np.allclose(g[:, :42], f[:, 42:] * sgn, atol=2e-6) and np.allclose(g[:, 42:], f[:, :42] * sgn, atol=2e-6)
print("отражение: Л′ = П с x → −x, П′ = Л с x → −x            ok")
print("всё ok")
