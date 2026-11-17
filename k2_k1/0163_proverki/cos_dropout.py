"""№ 4: где стоит LateDropout(0,8) относительно нормировки косинусной головы. Синтетика (ЦП): признак f ∈ R^384 (2·dim Sohn),
прототип w с cos(f, w) = c. Сейчас (sohn.py:729–731): cos(norm(drop(f)), norm(w)); вариант (б): <drop(norm(f)), norm(w)>.
Сравниваем среднее по маскам с оценкой без dropout (eval) и считаем потолок p(верный) при s = 16 и C классов."""
import json
import math
import torch
import torch.nn.functional as F

torch.manual_seed(0)
D, K = 384, 4000
out = {}
for kind in ("gauss", "swish"):
    f = torch.randn(D)
    if kind == "swish":
        f = F.silu(f * 1.5)                      # признаки после Dense + swish: несимметричные, часть около 0
    fh = F.normalize(f, dim=0)
    for c in (0.3, 0.6, 0.9):
        u = F.normalize(torch.randn(D) - (torch.randn(D) @ fh) * fh, dim=0)  # ⟂ f
        w = c * fh + math.sqrt(1 - c * c) * u
        drop = F.dropout(f.expand(K, D), 0.8, True)
        cur = F.normalize(drop, dim=1) @ w                           # сейчас
        dropn = F.dropout(fh.expand(K, D), 0.8, True)
        b = dropn @ w                                                # (б)
        out[f"{kind} c={c}"] = {"eval": round(float(fh @ w), 4), "сейчас_среднее": round(float(cur.mean()), 4),
                                "сейчас/eval": round(float(cur.mean() / (fh @ w)), 3), "(б)_среднее": round(float(b.mean()), 4),
                                "(б)/eval": round(float(b.mean() / (fh @ w)), 3), "(б)_sd": round(float(b.std()), 4)}
# потолок p(верный) при обучении: верный cos = ρ·1, прочие 0 (C − 1 штук); s = 16
for C in (33, 1001, 1034):
    for name, rho in (("сейчас", math.sqrt(0.2)), ("(б)", 1.0)):
        z = 16 * rho
        out[f"потолок p C={C} {name}"] = round(math.exp(z) / (math.exp(z) + C - 1), 3)
print(json.dumps(out, ensure_ascii=False, indent=1))
