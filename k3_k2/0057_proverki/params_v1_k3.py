"""к3 0057, В1: параметры Sohn (общий) — (А) кодировщик + головы вывода (без protos) против (Б) все. Сеть порта
SohnShared (fc5d15a) с SOHN_MODEL из диффа 0175 (dim 192, emb_dim 256) и T = 32 — как sohn_shared_cell 0178; param_counts —
mpk.model моей копии 7563b5e; классы — из чисел protos к2 (256 256 / 256 = 1 001 и т. д.). Для сравнения — родной SohnNet
(keras_param_count, trainable — как sohn_cell). Запуск: PYTHONPATH=<стабы>:<src> python params_v1_k3.py <корень дерева>"""
import sys
from pathlib import Path
root = Path(sys.argv[1]); sys.path[:0] = [str(root / "scripts")]
import torch  # noqa: E402
import turnir_sohn_shared as ts  # noqa: E402
from mpk import sohn  # noqa: E402
from mpk.model import param_counts  # noqa: E402

SOHN_MODEL = {"arch": "sohn_shared", "dim": 192, "emb_dim": 256, "emb_scale": 16.0}   # дифф 0175, строка 367
for ds, n in (("slovo", 1001), ("bukva", 33), ("uzsl", 682)):
    torch.manual_seed(0)
    m = ts.SohnShared({**SOHN_MODEL, "T": 32}, {ds: n}).eval()
    pc = param_counts(m)
    prot = sum(p.numel() for k, p in m.named_parameters() if "protos" in k.split("."))
    nat = sohn.keras_param_count(sohn.SohnNet(n, 192))["trainable"]
    print(f"{ds:6s} n_cls {n:5d}: total {pc['total']:>9,}  encoder {pc['encoder']:>9,}  inference_heads {pc['inference_heads']:>7,}  "
          f"train_only {pc['train_only']:>7,} (protos {prot:,} = n_cls·256: {prot == n * 256})  | (А) {(pc['encoder'] + pc['inference_heads']) / 1e6:.4f} M"
          f"  (Б) {pc['total'] / 1e6:.4f} M  | родной SohnNet {nat / 1e6:.4f} M".replace(",", " "))
    assert pc["train_only"] == prot == n * 256 and pc["total"] - pc["train_only"] == pc["encoder"] + pc["inference_heads"]
print("ВСЁ СОВПАЛО")
