"""ЦП: каждое из 10 предобучений группы 2 → resolve_init → файл весов; классификатор, который снимает ignore_weights их кода, на 1 034 класса.
Запуск: cd osnovnaya_model && CUDA_VISIBLE_DEVICES= python proverka_init.py"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "scripts")
import turnir_their as tt  # noqa: E402

R = Path("runs/TURNIR/train/pretrain")
for m, cls in (("SL-GCN", ("fc.weight", "fc.bias")), ("ST-GCN", ("fcn.weight", "fcn.bias"))):
    for s in range(5):
        w = tt.resolve_init(str(R / f"{m}_native/all/seed{s}"))
        sd = torch.load(w, map_location="cpu", weights_only=False)
        keys = {k.split("module.")[-1]: v for k, v in sd.items()}
        print(m, f"seed{s}", Path(w).relative_to(Path(R / f"{m}_native/all/seed{s}").resolve()).parts[0],
              "классификатор:", all(c in keys for c in cls), "классов:", keys[cls[0]].shape[0], "тензоров:", len(keys))
