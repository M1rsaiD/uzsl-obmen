"""к3 0046: независимая проверка № 2 (sohn.py ветки turnir-e2-levers, блоб 7a87d90) на ЦП, синтетика.
1) столбцы part_cols: при закреплённых центре и масштабе возмущение одной точки меняет только столбцы её части (все 32 сочетания (а′));
2) part_mask: «хоть одна точка не NaN», окно max_len, порядок токенов кисть Л, кисть П, [тело], лицо;
3) PartTokens: своя формула по model.py (МПК, A2) из тех же весов = выход модуля; пустой кадр — равные веса, без NaN.
Запуск: PYTHONPATH=<src> python parts_k3.py"""
import itertools, math
import numpy as np, torch
from mpk import sohn

rng = np.random.default_rng(20261117)
TOK = ("u02s", "u02c", "u01a", "u04", "u01b")
combos = [tuple(t for t in TOK if t in c) for r in range(len(TOK) + 1) for c in itertools.combinations(TOK, r)]
P_of = {}
n_checked = 0
orig = sohn.center_scale_u
for tokens in combos:
    if not tokens:
        continue  # без токенов — preprocess Sohn; № 2 в Э2 всегда со входом (а′)
    P = sohn.NUM_NODES_POSE if "u01a" in tokens else sohn.NUM_NODES
    pts = sohn.part_points(tokens)
    assert sorted(sum(pts, [])) == list(range(P)), tokens            # части покрывают все точки входа, без пересечений
    cols = sohn.part_cols(tokens)
    owner = {c: i for i, cs in enumerate(cols) for c in cs}
    x = rng.uniform(100, 600, (40, sohn.NUM_NODES_POSE, 2)).astype(np.float32)
    k, mean, std = orig(x, 16 / 9, tokens)
    sohn.center_scale_u = lambda *a, **kw: (k, mean, std)              # закрепляем нормировку: иначе любая точка меняет все столбцы
    try:
        f0 = sohn.preprocess_u(x, 16 / 9, tokens)
        for part, ps in enumerate(pts):
            for j in rng.choice(ps, size=min(3, len(ps)), replace=False):
                x1 = x.copy(); x1[:, j] += rng.normal(0, 5, (40, 2)).astype(np.float32)
                ch = np.flatnonzero(np.abs(sohn.preprocess_u(x1, 16 / 9, tokens) - f0).max(0) > 0)
                bad = [c for c in ch if owner[c] != part]
                assert ch.size and not bad, (tokens, part, j, bad[:5])
                n_checked += 1
    finally:
        sohn.center_scale_u = orig
print(f"1) столбцы: {len(combos) - 1} сочетаний (а′), {n_checked} возмущений точек — чужих столбцов 0")

# 2) маска
tokens = ("u01a", "u04")
x = rng.uniform(0, 1, (500, 122, 2)).astype(np.float32)
x[10:20, sohn.LH_LOC] = np.nan                         # кисть Л пропала
x[30, sohn.RH_LOC[1:]] = np.nan                        # у кисти П осталась одна точка — часть есть
x[40:45] = np.nan                                      # пустые кадры
m = sohn.part_mask(x, tokens)
assert m.shape == (sohn.MAX_LEN, 4)
assert m[10:20, 0].sum() == 0 and m[10:20, 1:].min() == 1 and m[30, 1] == 1 and m[40:45].sum() == 0
assert m.shape[1] == len(sohn.part_mask(x, ("u04",))[0]) + 1   # тело — только с U01a
print("2) маска: окно 384, «хоть одна точка», пустые кадры, тело только с U01a — верно")

# 3) PartTokens против своей формулы (model.py: rest_mlp(Linear → GELU(точный) → Linear) + part_emb; a = softmax(fuse_k(t)·q/√D), −1e4)
torch.manual_seed(0)
for tokens in (("u04", "u01b"), ("u01a", "u04", "u01b")):
    pt = sohn.PartTokens(tokens, 192).double()
    cols = sohn.part_cols(tokens)
    B, T = 3, 7
    feat = torch.randn(B, T, pt.n_in, dtype=torch.float64)
    mask = (torch.rand(B, T, pt.n_tok) > 0.3).double(); mask[0, 0] = 0             # кадр без частей
    out = pt(torch.cat([feat, mask], -1))
    toks = []
    for i, c in enumerate(cols):
        l1, l2 = pt.mlp[i][0], pt.mlp[i][2]
        h = feat[..., c] @ l1.weight.T + l1.bias
        h = 0.5 * h * (1 + torch.erf(h / math.sqrt(2)))                               # точный GELU
        toks.append(h @ l2.weight.T + l2.bias + pt.part_emb[i])
    t = torch.stack(toks, 2)
    s = (t @ pt.fuse_k.weight.T + pt.fuse_k.bias) @ pt.fuse_q / math.sqrt(192)
    s = torch.where(mask > 0, s, torch.full_like(s, -1e4))
    ref = (torch.softmax(s, 2)[..., None] * t).sum(2)
    err = float((out - ref).abs().max())
    assert err < 1e-10 and torch.isfinite(out).all(), err
    assert torch.allclose(out[0, 0], t[0, 0].mean(0)), "пустой кадр — не равные веса"
    n_par = sum(p.numel() for p in pt.parameters())
    print(f"3) PartTokens {tokens}: = своей формуле (макс. |Δ| {err:.1e}); пустой кадр = среднее токенов; параметров {n_par:,}")
print("ВСЁ СОВПАЛО")
