"""Скептик 3: независимая проверка находки «u02s: примеры ветки Sohn — выбросы масштаба, через BN части меняют нормировку
остальных». Только ЦП, только описательная статистика конвейера обучения (Stream → AugSet → collate_parts), ворота не считаются.
Запуск: CUDA_VISIBLE_DEVICES= python bn_check.py <seed_aug> <steps>"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/home/error_404/uzsign/osnovnaya_model"
sys.path.insert(0, ROOT + "/src")
sys.path.insert(0, ROOT + "/scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402

import m13_train as M  # noqa: E402
from mpk import sohn  # noqa: E402

torch.set_num_threads(8)
seed_aug, steps = int(sys.argv[1]), int(sys.argv[2])
recs, glosses, clips, (tr, va, te) = M.records("alluz", "all", inp=True)
train = tr + va
asp = recs.aspect
ARMS = {"ctrl": (), "u02s": ("u02s",), "u02c": ("u02c",), "u02sc": ("u02s", "u02c")}

# стемы: инициализация SohnNet при затравках 48, 1, 2 (stem — первый модуль) + обученные стемы Sohn (S+B, ALL-UZ)
stems = {}
for s in (48, 1, 2):
    torch.manual_seed(s)
    stems[f"init{s}"] = sohn.SohnNet(10, 192, channels=sohn.CHANNELS).stem.weight.detach().clone()
for nm, p in (("sb42", "runs/UzSohn/sb/all/seed42/last.pt"), ("alluz42", "runs/UzSohn/alluz_pre/all/seed42/last.pt")):
    stems[nm] = torch.load(f"{ROOT}/{p}", map_location="cpu")["model"]["stem.weight"].float().clone()


def z_rms(x, a, tok):
    """СКО-подобный размер z вокруг центра по осям (по конечным точкам), как в preprocess_u / preprocess."""
    if tok:
        k, mean, std = sohn.center_scale_u(x, a, tok)
        z = (x[:, :sohn.NUM_NODES] * k - mean) / std
    else:
        mean, std = sohn.center_scale(x[:, :sohn.NUM_NODES])
        z = (x[:, :sohn.NUM_NODES] - mean) / std
    with np.errstate(invalid="ignore"):
        return np.sqrt(np.nanmean(z[..., 0] ** 2)), np.sqrt(np.nanmean(z[..., 1] ** 2))


out = {arm: {nm: [] for nm in stems} for arm in ARMS}
rms = {arm: {"br": [], "nb": []} for arm in ARMS}
nbr, nall, eq_checked, eq_ok = [], 0, 0, 0
cv_acc = {arm: {nm: [] for nm in stems} for arm in ARMS}   # var_all по частям — для естественного шума BN
cvk_acc = {arm: {nm: [] for nm in stems} for arm in ARMS}  # var без ветки по частям
for step, batch in enumerate(M.Stream(train, steps, seed_aug, 512, 32768)):
    n = len(batch) // M.PARTS
    for p in range(M.PARTS):
        ch = batch[p * n:(p + 1) * n]
        feats = {arm: [] for arm in ARMS}
        br = []
        for (i, pos) in ch:
            x122, ab = recs[i]
            xa = sohn.augment(x122, ab, np.random.default_rng([seed_aug, pos]))
            a0, b0 = recs.off[i], recs.off[i + 1]
            x118 = sohn.augment(recs.xy[a0:b0], recs.absent[a0:b0], np.random.default_rng([seed_aug, pos]))
            b = not M.shoulders_seen(xa)
            br.append(b)
            for arm, tok in ARMS.items():
                f = sohn.preprocess(x118) if not tok else sohn.preprocess_u(xa, float(asp[i]), tok)
                feats[arm].append(f)
                r = z_rms(x118 if not tok else xa, float(asp[i]), tok)
                rms[arm]["br" if b else "nb"].append(r)
            if b and eq_checked < 400:   # вход ветки Sohn у u02s = вход контроля бит в бит
                eq_checked += 1
                eq_ok += int(np.array_equal(feats["u02s"][-1], feats["ctrl"][-1]))
        br = np.array(br)
        nbr.append(int(br.sum()))
        nall += len(br)
        for arm in ARMS:
            X, Mk = sohn.collate(feats[arm])
            X = X * Mk[..., None]
            w = Mk[..., None].float()
            keep = torch.from_numpy(~br)
            for nm, W in stems.items():
                H = X @ W.T
                nA = w.sum()
                mA = (H * w).sum((0, 1)) / nA
                vA = (((H - mA) ** 2) * w).sum((0, 1)) / nA
                Hk, wk = H[keep], w[keep]
                nK = wk.sum()
                mK = (Hk * wk).sum((0, 1)) / nK
                vK = (((Hk - mK) ** 2) * wk).sum((0, 1)) / nK
                ratio = (vA / vK).numpy()
                shift = ((mA - mK).abs() / vK.sqrt()).numpy()
                # нуль: убрать столько же СЛУЧАЙНЫХ примеров не из ветки
                nb_idx = np.flatnonzero(~br)
                drop = np.random.default_rng([step, p]).choice(nb_idx, size=int(br.sum()), replace=False)
                keepR = np.ones(len(br), bool)
                keepR[drop] = False
                Hr, wr = H[torch.from_numpy(keepR)], w[torch.from_numpy(keepR)]
                mR = (Hr * wr).sum((0, 1)) / wr.sum()
                vR = (((Hr - mR) ** 2) * wr).sum((0, 1)) / wr.sum()
                ratio_null = float(np.median((vA / vR).numpy()))
                cvk_acc[arm][nm].append(vK.numpy())
                # примеры ветки: средний квадрат выхода stem на шаг (отн. остальных)
                e2 = ((H ** 2) * w).sum((1, 2)) / w.sum((1, 2)) / H.shape[-1]
                e2b = float(e2[~keep].mean()) / float(e2[keep].mean()) if (~keep).any() else float("nan")
                tshare = float(w[~keep].sum() / nA)
                out[arm][nm].append({"n_br": int(br.sum()), "ratio_med": float(np.median(ratio)), "ratio_max": float(ratio.max()),
                                     "ratio_null": ratio_null,
                                     "shift_med": float(np.median(shift)), "e2_br_over_nb": e2b, "t_share": tshare})
                cv_acc[arm][nm].append(vA.numpy())
    print(f"шаг {step + 1}/{steps}: ветка Sohn {sum(nbr)}/{nall}", flush=True)

res = {"seed_aug": seed_aug, "steps": steps, "parts": len(nbr), "examples": nall, "branch_total": int(sum(nbr)),
       "branch_share": sum(nbr) / nall, "branch_per_part_p10_p50_p90": np.percentile(nbr, [10, 50, 90]).tolist(),
       "eq_checked": eq_checked, "eq_bit_exact": eq_ok, "arms": {}}
for arm in ARMS:
    a = {}
    for k in ("br", "nb"):
        v = np.array(rms[arm][k])
        a[f"rms_{k}_x_med"], a[f"rms_{k}_y_med"] = float(np.nanmedian(v[:, 0])), float(np.nanmedian(v[:, 1]))
        a[f"rms_{k}_nan"] = int(np.isnan(v).any(1).sum())
    a["rms_ratio_x"] = a["rms_br_x_med"] / a["rms_nb_x_med"]
    a["rms_ratio_y"] = a["rms_br_y_med"] / a["rms_nb_y_med"]
    for nm in stems:
        L = out[arm][nm]
        r = np.array([d["ratio_med"] for d in L])
        V = np.stack(cv_acc[arm][nm])   # [части, 192]
        VK = np.stack(cvk_acc[arm][nm])
        S = np.sqrt(VK / V)             # множитель нормировки обычных примеров «с веткой / без неё» (часть, канал)
        a[nm] = {"ratio_med_p10_p50_p90": np.percentile(r, [10, 50, 90]).round(4).tolist(), "ratio_med_max": float(r.max()),
                 "ratio_chan_max_p50": float(np.median([d["ratio_max"] for d in L])),
                 "shift_med_p50": float(np.median([d["shift_med"] for d in L])),
                 "e2_br_over_nb_p50": float(np.nanmedian([d["e2_br_over_nb"] for d in L])),
                 "t_share_p50": float(np.median([d["t_share"] for d in L])),
                 "cv_var_across_parts_med": float(np.median(V.std(0) / V.mean(0))),
                 "cv_var_wo_branch_across_parts_med": float(np.median(VK.std(0) / VK.mean(0))),
                 "scale_normal_mean_med": float(np.median(S.mean(0))), "scale_normal_cv_med": float(np.median(S.std(0) / S.mean(0))),
                 "ratio_null_p10_p50_p90": np.percentile([d["ratio_null"] for d in L], [10, 50, 90]).round(4).tolist(),
                 "corr_nbr_ratio": float(np.corrcoef([d["n_br"] for d in L], r)[0, 1])}
    res["arms"][arm] = a
path = f"/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/s3_u02bn/bn_check_seed{seed_aug}_steps{steps}.json"
json.dump(res, open(path, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
