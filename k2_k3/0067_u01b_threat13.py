"""Этап 2 UzSohn, угроза 13 у U01b (к3 0040: вопрос 6 и просьба 2) — справочно, ЦП, не ворота и не блок хеша.
1–2: как revision_r4/u04_stem_energy_r4.py и u04_awp_ratio_r4.py, вход плеча этапа 2 — токен u01b (708 + 84 столбца).
3: все записи ALL-UZ без аугментации — доля значений U01b, обрезанных на ±10, и доля кистей без ℓ_h (42 нуля).
Код — ветка uzsohn-stage2-u01b (worktree к2), данные — OM (только чтение). Запуск: CUDA_VISIBLE_DEVICES= nice -n 19 python <этот файл>."""
import json, sys
from pathlib import Path
import numpy as np, torch
W = Path("/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/wt_u01b")
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(W / "src"), str(W / "scripts"), str(OM.parent / "uzsign_e1/src"), str(OM.parent / "uzsign_ai_v2/src")]
import m13_train as T
from mpk import sohn
assert "u01b" in sohn.INPUT_TOKENS and Path(sohn.__file__).resolve().is_relative_to(W.resolve())
torch.set_num_threads(4)
TOK, NB, NU = ("u01b",), 708, sohn.HAND_LOCAL
assert sohn.input_channels(()) == NB and sohn.input_channels(TOK) == NB + NU == 792
res = {"код": "uzsohn-stage2-u01b (worktree к2)", "токены": TOK}

# 1. Доля прежнего пути после BN при инициализации (как u04_stem_energy_r4.py: 2 000 записей, default_rng(5), manual_seed(42))
recs, gl, clips, _ = T.records("alluz", None, inp=True)
idx = np.random.default_rng(5).choice(len(recs), 2000, replace=False)
F = np.concatenate([sohn.preprocess_u(recs[int(i)][0], float(recs.aspect[i]), TOK) for i in idx])
v = F.var(0)
torch.manual_seed(42)
Wt = sohn.SohnNet(3531, channels=sohn.input_channels(TOK)).stem.weight.detach().numpy()   # [192, 792]
ho, ht = F[:, :NB] @ Wt[:, :NB].T, F @ Wt.T
r = float(np.median(ho.var(0) / ht.var(0)))
res["stem при инициализации"] = {"доля Σ Var у 84 столбцов U01b": round(float(v[NB:].sum() / v.sum()), 3),
                                 "Var(прежний путь)/Var(stem), медиана по 192 каналам": round(r, 3),
                                 "амплитуда прежнего пути после BN": round(r ** 0.5, 3), "записей": 2000}

# 2. AWP: доля градиента stem на прежних 708 столбцах (как u04_awp_ratio_r4.py: фолд №4 по порядку, 3 пакета по 64 из обучения;
#    rng — свой default_rng(0) на каждый вариант, поэтому пакеты у «с нуля» и «база» одни и те же)
fold = sorted(json.loads(T.SPLIT["alluz"].read_text())["folds"])[4]
recs, gl, clips, (tr, va, te) = T.records("alluz", fold, inp=True)
names = sorted(set(gl)); lab = {g: k for k, g in enumerate(names)}
labels = np.array([lab[g] for g in gl])
awp = {}
for init in (None, "base"):
    rng = np.random.default_rng(0)
    recs.tokens = TOK
    ds = T.AugSet(recs, labels, 42, True)
    torch.manual_seed(42)
    m = sohn.SohnNet(len(names), channels=sohn.input_channels(TOK)).train()
    if init == "base":   # прежние 708 столбцов и всё остальное — веса предобучения базы (S+B), новые 84 — glorot
        sd = torch.load(OM / "runs/UzSohn/sb/all/seed42/best.pt", map_location="cpu")["model"]
        w = m.stem.weight.detach().clone(); w[:, :NB] = sd["stem.weight"]
        sd = {k: x for k, x in sd.items() if not k.startswith("classifier.")}; sd["stem.weight"] = w
        m.load_state_dict(sd, strict=False)
    rs = []
    for b in range(3):
        bi = rng.choice(tr, 64, replace=False)
        items = [ds[(int(i), int(rng.integers(10**6)))] for i in bi]
        x, mk = sohn.collate([a for a, _ in items]); y = torch.tensor([b_ for _, b_ in items])
        m.zero_grad()
        torch.nn.functional.cross_entropy(m(x, mk), y, label_smoothing=0.1).backward()
        g = m.stem.weight.grad
        assert g.shape[1] == NB + NU
        old, new = g[:, :NB].norm().item(), g[:, NB:].norm().item()
        rs.append((round(old / g.norm().item(), 3), round(new / old, 3)))
    awp[init or "с нуля"] = {"‖g_прежн‖/‖g_stem‖": [a for a, _ in rs], "‖g_нов‖/‖g_прежн‖": [b for _, b in rs]}
res["AWP, фолд №4, 3 пакета по 64"] = awp

# 3. Все записи ALL-UZ без аугментации: обрезка ±10 и кисти без ℓ_h; своя формула (3.1) сверяется с выходом hand_local
recs, gl, clips, _ = T.records("alluz", None, inp=True)
n_val = n_clip = n_all = n_clip_all = 0
no_l = {"Л": 0, "П": 0}; palm0 = 0; absz = []
for i in range(len(recs)):
    x, a = recs[i][0], float(recs.aspect[i])
    Fu = sohn.preprocess_u(x, a, TOK)[:, NB:]
    xs = x[:sohn.MAX_LEN, :sohn.NUM_NODES] * np.array([1.0, a], np.float32)
    mine = []
    for side, H in (("Л", sohn.LH_LOC), ("П", sohn.RH_LOC)):
        q = xs[:, H]
        palm = np.linalg.norm(q[:, 9] - q[:, 0], axis=-1)
        med = np.nanmedian(palm) if np.isfinite(palm).any() else np.nan
        if not np.isfinite(med) or med == 0:
            no_l[side] += 1; palm0 += bool(np.isfinite(med) and med == 0)
            mine.append(np.zeros((len(q), 42), np.float32)); continue
        with np.errstate(invalid="ignore"):
            z = (q - q[:, :1]) / med
        ok = np.isfinite(z)
        n_val += ok.sum(); n_clip += (np.abs(z[ok]) > sohn.EXTRA_CLIP).sum(); absz.append(np.abs(z[ok]))
        mine.append(np.clip(np.nan_to_num(z, nan=0.0), -sohn.EXTRA_CLIP, sohn.EXTRA_CLIP).reshape(len(q), -1))
    assert np.allclose(np.concatenate(mine, 1), Fu, rtol=1e-5, atol=1e-5), i
    n_all += Fu.size; n_clip_all += (np.abs(Fu) >= sohn.EXTRA_CLIP).sum()
az = np.concatenate(absz); N = len(recs)
res["ALL-UZ без аугментации"] = {
    "записей": N, "кистей без ℓ_h (42 нуля), доля от 2·N": round((no_l["Л"] + no_l["П"]) / (2 * N), 4),
    "из них Л / П": [no_l["Л"], no_l["П"]], "из них ладонь = 0 (кадры с 0 и 9 есть)": palm0,
    "обрезанных на ±10 среди значений кистей с ℓ_h, где есть точка и запястье (число / из / доля)": [int(n_clip), int(n_val), float(n_clip / n_val)],
    "обрезанных на ±10 среди всех значений U01b, кадры × 84 (число / из / доля)": [int(n_clip_all), int(n_all), float(n_clip_all / n_all)],
    "|z| у таких значений: p50 / p99 / p99,9 / max": [round(float(np.quantile(az, q)), 3) for q in (0.5, 0.99, 0.999)] + [round(float(az.max()), 2)]}
print(json.dumps(res, ensure_ascii=False, indent=1))
