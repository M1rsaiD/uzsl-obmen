"""Входы модели риска (risk_series_k2.py) по S-кодам — агрегаты для сверки консультантами (без названий фолдов)."""
import json
import numpy as np
R = "/home/error_404/uzsign/osnovnaya_model/runs/UzSohn/alluz_pre"
F = sorted(json.load(open("/home/error_404/uzsign/uzsign_ai_v2/splits/uzsl_allz_loso.json"))["folds"])
code = json.load(open("/home/error_404/uzsign/uzsign_ai_v2/data/signer_codes.local.json"))["signers"]
A = np.array([[100 * json.load(open(f"{R}/{f}/seed{s}/metrics.json"))["top1"] for f in F] for s in (42, 43, 44)])
N = np.array([json.load(open(f"{R}/{f}/seed42/metrics.json"))["n_test"] for f in F])
NV = np.array([143, 168, 479, 1158, 621, 135, 265, 812, 76, 77])
out = {code[f]: {"sd_seeds_pp": round(float(A[:, i].std(ddof=1)), 4), "top1_mean_pp": round(float(A[:, i].mean()), 4),
                 "n_test": int(N[i]), "n_as_Vk": int(NV[i])} for i, f in enumerate(F)}
print(json.dumps({"order": [code[f] for f in F], "macro_by_seed_pp": [round(float(x), 4) for x in A.mean(1)],
                  "by_signer": out}, ensure_ascii=False, indent=1))
