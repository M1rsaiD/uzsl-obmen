"""Синтетика (без данных и моделей): ворота U01a — «место/плечи» и «место/лицо» с общим началом (нос) различаются только
делителем. Если ширина плеч / ширина лица = const, процентили пар совпадают → «только плечи» = 0 → q = 0.
Функции ворот — из scripts/uzsohn_gates.py (place_feats, pct_ref, pct) без изменений."""
import sys
from pathlib import Path

import numpy as np

OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(OM / "scripts")]
import uzsohn_gates as G  # noqa: E402
from mpk import sohn  # noqa: E402

L = sohn.LOCAL


class FakeRecs:
    def __init__(self, xs, asp):
        self.xs, self.aspect = xs, asp

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        return self.xs[i], None


def make(cv_ratio, n_cls=200, per=6, T=20, seed=0):
    rng = np.random.default_rng(seed)
    place = rng.normal(0, 0.6, (n_cls, 2))                  # место класса (в ширинах плеч)
    xs, lab = [], []
    for c in range(n_cls):
        for _ in range(per):
            x = np.full((T, 122, 2), np.nan, np.float32)
            sw = rng.uniform(0.25, 0.4)                     # ширина плеч записи (доли кадра)
            r = 2.7 * np.exp(rng.normal(0, cv_ratio))       # ширина плеч / ширина лица
            fw = sw / r
            nose = np.array([0.5, 0.3]) + rng.normal(0, 0.02, 2)
            x[:, L[1]] = nose
            x[:, L[33]] = nose + [-fw / 2, -0.02]
            x[:, L[263]] = nose + [fw / 2, -0.02]
            x[:, 118] = nose + [sw / 2, 0.15]
            x[:, 119] = nose + [-sw / 2, 0.15]
            w = nose + (place[c] + rng.normal(0, 0.15, 2)) * sw + rng.normal(0, 0.005, (T, 2))
            for k, p in enumerate(sohn.RH_LOC):
                x[:, p] = w + 0.01 * np.array([np.cos(k), np.sin(k)]) * (k > 0)
            xs.append(x)
            lab.append(c)
    return FakeRecs(xs, np.ones(len(xs), np.float32)), lab


for cv in (0.0, 0.05, 0.10):
    recs, lab = make(cv)
    F = G.place_feats(recs)
    by = {}
    for i, c in enumerate(lab):
        by.setdefault(c, []).append(F[i])
    means = [{c: np.mean([v[k] for v in vs], 0) for c, vs in by.items()} for k in range(3)]
    rng = np.random.default_rng(G.RNG)
    rp = G.ref_pairs(len(means[0]), rng); refs = [G.pct_ref(m, rp) for m in means]
    cls = sorted(by)
    place = only = 0
    for a in cls:
        for b in cls:
            if a == b:
                continue
            pp_s = G.pct(refs[1], float(np.linalg.norm(means[1][a] - means[1][b])))
            pp_f = G.pct(refs[2], float(np.linalg.norm(means[2][a] - means[2][b])))
            if pp_s >= G.PCT:
                place += 1
                only += pp_f < G.PCT
    print(f"КВ(плечи/лицо) = {cv:.2f}: пар «место» {place}, из них «только плечи» {only}, q = {only / place:.4f} (порог {G.Q_MIN})")
