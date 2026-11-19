"""к1 0155 разд. 2: diff четырёх шаблонов предобучения S+B (turnir_train.pre_tpl) без блока model — после with_defaults, как у раннера."""
import sys
from pathlib import Path
import yaml
WT = Path(sys.argv[1])  # worktree с scripts/turnir_train.py
RUNS = Path.home() / "uzsign/osnovnaya_model/runs"
sys.path[:0] = [str(WT / "scripts"), str(WT / "src")]
import turnir_train as tt  # noqa: E402
M = {"A1": "M8/m83_pretrain_sb_a1", "A2": "M9/pretrain_slovo_bukva_a2", "MPK_large": "M8/m83_pretrain_sb_a1_big",
     "SL-GCN (= ST-GCN)": "M11P/pretrain_sb_slgcn_orig"}
flat = lambda d, k="": {kk: v for x, y in d.items() for kk, v in flat(y, f"{k}{x}.").items()} if isinstance(d, dict) else {k[:-1]: d}
F = {}
for m, p in M.items():
    c = tt.with_defaults(yaml.safe_load((RUNS / p / "all/seed0/config.yaml").read_text()))
    F[m] = {k: v for k, v in flat(c).items() if not k.startswith("model.")}
keys = sorted(set().union(*F.values()))
diff = [k for k in keys if len({repr(F[m].get(k, "<нет>")) for m in M}) > 1]
print(f"полей вне model: {len(keys)}; различаются: {len(diff)}")
for k in diff:
    print(k, " | ".join(f"{m}: {F[m].get(k, '<нет>')}" for m in M))
show = ["train.lr", "train.weight_decay", "train.warmup_frac", "train.steps", "train.batch_size", "train.ema", "train.ema_decay",
        "train.eval_every", "train.select", "finetune.freeze_steps", "finetune.unfreeze_every", "finetune.backbone_lr_mult",
        "loss.lambda_emb", "loss.beta_part", "loss.label_smoothing"]
print("--- общие (значение A1):")
for k in show:
    print(k, F["A1"].get(k, "<нет>"), "(одинаково у всех)" if k not in diff else "(РАЗЛИЧАЕТСЯ)")
