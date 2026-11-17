"""Цена/выгода двух добавленных рычагов (R3: места 6–7) — к голосу к2 по вопросу 1. ЦП. Модель шума и правило — как mc_check.py."""
import numpy as np
THR, MINB = 1.0081957667939179, 6
SIG = {"base": [2.92, 2.63, 3.13, 3.17, 3.04, 1.33, 1.36, 1.40, 1.44, 2.09],
       "sohn": [2.20, 1.98, 2.35, 2.38, 2.28, 1.05, 1.08, 1.11, 1.14, 1.65]}
N = 400_000
def acc(d): return (d.mean(1) >= THR) & ((d > 0).sum(1) >= MINB)
def chain(sig, mus, rng, n=N):
    sig = np.asarray(sig); cur = rng.standard_normal((n, 10)) * sig; take = []
    for mu in mus:
        mu = np.broadcast_to(np.asarray(mu, float), (10,))
        e = rng.standard_normal((n, 10)) * sig + mu
        ok = acc(e - cur); take.append(ok.mean()); cur = np.where(ok[:, None], e, cur)
    return np.array(take)
rng = np.random.default_rng(7)
M7 = [1.5] * 5 + [0.3] * 5          # профиль A-EMA из M7 (Slovo +1,5, Bukva +0,3) — только как форма эффекта, не для выбора
for name, sig in SIG.items():
    z = chain(sig, [0] * 7, rng)
    print(f"== {name}: P(нулевой принят) поз.1..7 = " + " ".join(f"{x:.3f}" for x in z))
    for mu in (1.0, 1.5, 2.0):
        t = chain(sig, [0] * 5 + [mu, mu], rng)
        q = z[5] / (z[5] + t[5])      # безубыточность: q·P(μ на поз.6) = (1−q)·P(0 на поз.6)
        print(f"  μ=+{mu}: поз.6 {t[5]:.3f} поз.7 {t[6]:.3f} | безубыточная априорная доля q* (поз.6) = {q:.3f}")
    t = chain(sig, [0] * 5 + [M7], rng); t1 = chain(sig, [M7], rng)
    print(f"  профиль M7 (S +1,5 / B +0,3, Δ̄ = 0,9 < THR): поз.1 {t1[0]:.3f}, поз.6 {t[5]:.3f}")
    # порядок «рычаги 0059 первыми»: два рычага μ=+1 на местах 1–2, затем 5 нулевых (а′); и наоборот
    first = chain(sig, [1.0, 1.0] + [0] * 5, rng); last = chain(sig, [0] * 5 + [1.0, 1.0], rng)
    print(f"  μ=+1 на местах 1–2: {first[0]:.3f}/{first[1]:.3f}; на местах 6–7: {last[5]:.3f}/{last[6]:.3f}")
    a_first = chain(sig, [0, 0] + [1.0] + [0] * 4, rng); a_orig = chain(sig, [1.0] + [0] * 6, rng)
    print(f"  рычаг (а′) с μ=+1 на 1-м месте цепочки (а′): {a_orig[0]:.3f}; если два рычага 0059 (нулевые) впереди — {a_first[2]:.3f}")
