"""Скептик ред. 3 (линза «код»), ветка d7b0d68, ЦП, синтетика: (1) столбец — в часть своей точки (возмущение точки → какие
столбцы preprocess_u меняются) для всех 8 сочетаний u01a/u04/u01b, без u02 и с u02s,u02c; (2) тождественные нули входа;
(3) part_mask: длина = длине preprocess_u (T < 384, T > 384), отражение меняет кисти местами; (4) состав частей (21/21/[4]/76).
Запуск: CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 python -I -B cols_mask_check.py <worktree>"""
import itertools
import sys

WT = sys.argv[1]
sys.path[:0] = [WT + "/src"]
import numpy as np  # noqa: E402
from mpk import sohn  # noqa: E402

COMBOS = [tuple(t for t, on in zip(("u01a", "u04", "u01b"), b) if on) for b in itertools.product((0, 1), repeat=3)]
rng = np.random.default_rng(1)
T = 30
x0 = rng.uniform(0.2, 0.8, (T, sohn.NUM_NODES_POSE, 2)).astype(np.float32)
bad = 0
for extra in ((), ("u02s", "u02c")):
    for tok0 in COMBOS:
        tok = tuple(t for t in sohn.INPUT_TOKENS if t in tok0 + extra)
        if not tok:
            continue
        pts = sohn.part_points(tok)
        cols = sohn.part_cols(tok)
        owner = {c: k for k, cs in enumerate(cols) for c in cs}
        f0 = sohn.preprocess_u(x0, 0.5625, tok)
        assert f0.shape[1] == sohn.input_channels(tok)
        P = sohn.NUM_NODES_POSE if "u01a" in tok else sohn.NUM_NODES
        foreign = 0
        for k, ps in enumerate(pts):
            for p in ps:
                x = x0.copy()
                x[:, p] += rng.normal(0, 0.05, (T, 2)).astype(np.float32)
                ch = np.flatnonzero(np.abs(sohn.preprocess_u(x, 0.5625, tok) - f0).max(0) > 0)
                foreign += sum(owner[c] != k for c in ch)
        # u02*: центр/масштаб от плеч (точки 118, 119) — возмущение плеча при u02* без u01a
        if "u01a" not in tok:
            x = x0.copy()
            x[:, sohn.NUM_NODES] += 0.05
            ch_sh = np.flatnonzero(np.abs(sohn.preprocess_u(x, 0.5625, tok) - f0).max(0) > 0)
        else:
            ch_sh = []
        bad += foreign
        print(f"{'+'.join(tok):24s} части {[len(p) for p in pts]} столбцов {[len(c) for c in cols]} чужих столбцов {foreign}"
              + (f"; плечо 11 (не во входе) меняет {len(ch_sh)} столбцов" if len(ch_sh) else ""))
# (2) тождественные нули на 200 случайных записях
for tok in (("u01a", "u04", "u01b"), ("u04",), ("u01b",)):
    Z = None
    for s in range(200):
        r = np.random.default_rng(100 + s)
        x = r.uniform(0.0, 1.0, (int(r.integers(5, 60)), sohn.NUM_NODES_POSE, 2)).astype(np.float32)
        nz = (np.abs(sohn.preprocess_u(x, 0.5625, tok)) > 0).any(0)
        Z = nz if Z is None else (Z | nz)
    z = np.flatnonzero(~Z)
    off = 6 * (sohn.NUM_NODES_POSE if "u01a" in tok else sohn.NUM_NODES) + (sohn.HAND_FEATS if "u04" in tok else 0)
    print(f"нули {'+'.join(tok)}: абсолютные {z.tolist()}; относительно начала U01b: {(z - off).tolist() if 'u01b' in tok else '-'}")
# (3) маска: длина и окно
for TT in (10, 384, 500):
    x = rng.uniform(0.2, 0.8, (TT, sohn.NUM_NODES_POSE, 2)).astype(np.float32)
    x[:5, sohn.LH_LOC] = np.nan
    for tok in ((), ("u01a", "u04", "u01b")):
        f = sohn.preprocess_u(x, 0.5625, tok) if tok else sohn.preprocess(x[:, :sohn.NUM_NODES])
        m = sohn.part_mask(x if tok else x[:, :sohn.NUM_NODES], tok)
        print(f"T={TT} tok={tok}: preprocess {f.shape[0]} строк, part_mask {m.shape}, кисть Л отсутствует в кадрах {np.flatnonzero(m[:, 0] == 0).tolist()[:8]}")
# отражение: augment c флипом — маска кистей по x после аугментации
x = rng.uniform(0.2, 0.8, (40, sohn.NUM_NODES, 2)).astype(np.float32)
x[:, sohn.LH_LOC] = np.nan
ab = np.zeros((40, 4), bool)
seen = set()
for s in range(60):
    xa = sohn.augment(x, ab, np.random.default_rng([s, 1]))
    m = sohn.part_mask(xa)
    seen.add((int(m[:, 0].max()), int(m[:, 1].max())))
print("маска кистей (Л, П) после augment, когда во входе нет кисти Л:", sorted(seen), "(1, 0) = отражение поменяло кисти местами")
print("ИТОГ: чужих столбцов всего", bad)
