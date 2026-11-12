"""к3: независимый пересчёт q ворот U01a на синтетике (свой код, без uzsohn_gates / sohn).
Модель данных — как описано в 0143 / r1 (место класса в ширинах плеч, ширина плеч U(0,25; 0,4), отношение
плечи/лицо = 2,7·exp(N(0, КВ))), плюс варианты, которых у к2 нет: покадровый шум ширины плеч, отношение на уровне
исполнителя, 2 записи на класс. Ворота: «место» — процентиль |Δ места/плечи| ≥ 50 (форма у всех одна — условие формы
опущено, как у к2), «только плечи» — процентиль |Δ места/лицо| < 50; опорные 20 000 пар — общие для двух пространств."""
import sys
import numpy as np

def make(cv, n_cls=200, per=6, T=20, seed=0, frame_sw_cv=0.0, signer_level=False, n_sig=10):
    g = np.random.default_rng(seed)
    place = g.normal(0, 0.6, (n_cls, 2))
    sig_r = 2.7 * np.exp(g.normal(0, cv, n_sig))            # отношение — свойство исполнителя
    ps, pf, lab = [], [], []
    for c in range(n_cls):
        for _ in range(per):
            sw = g.uniform(0.25, 0.4)
            r = sig_r[g.integers(n_sig)] if signer_level else 2.7 * np.exp(g.normal(0, cv))
            fw = sw / r
            nose = np.array([0.5, 0.3]) + g.normal(0, 0.02, 2)
            w = nose + (place[c] + g.normal(0, 0.15, 2)) * sw + g.normal(0, 0.005, (T, 2))
            v = w - nose                                       # запястье − нос, [T, 2]
            sw_t = sw * np.exp(g.normal(0, frame_sw_cv, T))    # покадровая ширина плеч (поворот корпуса)
            ps.append((v / sw_t[:, None]).mean(0))
            pf.append((v / fw).mean(0))
            lab.append(c)
    return np.array(ps), np.array(pf), np.array(lab)

def q_gate(ps, pf, lab, seed_ref=20261111, weight=None, tau=None):
    cls = np.unique(lab)
    ms = np.array([ps[lab == c].mean(0) for c in cls]); mf = np.array([pf[lab == c].mean(0) for c in cls])
    g = np.random.default_rng(seed_ref)
    rp = g.integers(0, len(cls), (20000, 2)); rp = rp[rp[:, 0] != rp[:, 1]]
    rs = np.sort(np.linalg.norm(ms[rp[:, 0]] - ms[rp[:, 1]], axis=1))
    rf = np.sort(np.linalg.norm(mf[rp[:, 0]] - mf[rp[:, 1]], axis=1))
    a, b = np.where(~np.eye(len(cls), dtype=bool))
    p_s = 100 * np.searchsorted(rs, np.linalg.norm(ms[a] - ms[b], axis=1)) / len(rs)
    p_f = 100 * np.searchsorted(rf, np.linalg.norm(mf[a] - mf[b], axis=1)) / len(rf)
    place = p_s >= 50
    only = place & (p_f < 50)
    w = np.ones(len(a)) if tau is None else np.exp(-(p_s - 50) / tau)   # сгущение ошибок к порогу
    q = (w * only).sum() / (w * place).sum()
    near = place & (p_s - 50 < 0.1)
    return q, place.sum(), only.sum(), (only[near].mean() if near.any() else np.nan), near.sum()

if __name__ == "__main__":
    print("1) как у к2 (r1): 200 классов × 6 записей, затравка 0, равномерно")
    for cv in (0.0, 0.05, 0.10):
        q, P, O, qn, nn = q_gate(*make(cv))
        print(f"   КВ {cv:.2f}: «место» {P}, «только плечи» {O}, q = {q:.4f}; у порога (<0,1 процентиля) доля {qn:.2f} из {nn}")
    print("2) 100 реализаций (затравки 1..100), КВ 0,05: медиана / максимум q")
    for lbl, kw, tau in (("6 зап., равномерно", {}, None), ("2 зап., равномерно", dict(per=2), None),
                         ("6 зап., τ = 2", {}, 2.0), ("2 зап., τ = 2", dict(per=2), 2.0), ("2 зап., τ = 1", dict(per=2), 1.0),
                         ("6 зап., покадр. шум плеч 0,10", dict(frame_sw_cv=0.10), None),
                         ("2 зап., отношение — исполнителя (10), τ = 2", dict(per=2, signer_level=True), 2.0)):
        qs = np.array([q_gate(*make(0.05, seed=s, **kw), tau=tau)[0] for s in range(1, 101)])
        print(f"   {lbl}: медиана {np.median(qs):.3f}, максимум {qs.max():.3f}, q ≥ 0,5: {(qs >= 0.5).sum()} из 100")
