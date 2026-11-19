"""к3 0054 (сверка порта группы 3 0175 по 4.7): turnir_sohn_shared.py (блоб fc5d15a) — на ДЕРЕВЕ РЫЧАГОВ (sohn.py 7f1ac1f,
m13_train.py 5f83fec): при (а) группа 3 идёт после перемотки, то есть на этом коде; тесты 0175 шли на e516900 (без рычагов).
mpk.* — моя копия mpe-slovo-bukva 7563b5e (≠ f6a5a72 на ПК); m13_build и uzsign_e1.train — заглушки; синтетика, ЦП.
Цепочка как у группы 3, но предобучение — один набор (S+B с двумя головами моя копия mpk не умеет) → дообучение Slovo\nс init_from, заморозкой и разморозкой сверху вниз.
Запуск (не -I): PYTHONPATH=<стабы>:<src> python g3_lev_k3.py <корень дерева>"""
import json, sys, tempfile
from pathlib import Path
import numpy as np, torch
ROOT = Path(sys.argv[1]); sys.path.insert(0, str(ROOT / "scripts"))
import turnir_sohn_shared as P
from mpk import sohn
from mpk.train import param_groups

tmp = Path(tempfile.mkdtemp()); P.M13_DIR = tmp / "M13"; P.M13_DIR.mkdir()
rng = np.random.default_rng(0)
def make(name, n_cls, n_per, signers):   # кэш M13, кэш МПК (H/W), разбиение fold1: train / val / test — разные исполнители
    clips, xs = [], []
    for c in range(n_cls):
        for k in range(n_per):
            L = int(rng.integers(20, 60)); x = rng.random((L, sohn.NUM_NODES, 2)).astype(np.float32)
            x[rng.random(L) < 0.3, :21] = np.nan   # пропуски кисти
            xs.append(x); clips.append({"sample_id": f"{name}_{c}_{k}", "signer_id": signers[k % len(signers)], "gloss": f"g{c}"})
    off = np.concatenate([[0], np.cumsum([len(x) for x in xs])])
    np.savez(P.M13_DIR / f"{name}.npz", xy=np.concatenate(xs), absent=np.zeros((off[-1], 4), bool), offsets=off, clips=json.dumps(clips))
    np.savez(tmp / f"{name}_mpk.npz", clips=json.dumps([{"sample_id": c["sample_id"]} for c in clips]), aspect=np.full(len(clips), 1.3))
    part = {"train": signers[:-2], "val": signers[-2:-1], "test": signers[-1:]}
    fold = {r: [c["sample_id"] for c in clips if c["signer_id"] in s] for r, s in part.items()}
    (tmp / f"{name}_split.json").write_text(json.dumps({"folds": {"fold1": fold}}))
    pre = {"train": fold["train"] + fold["test"], "val": fold["val"], "test": []}   # S+B: без теста (has_test = False)
    (tmp / f"{name}_split_pre.json").write_text(json.dumps({"folds": {"fold1": pre}}))
    return {"dataset": name, "cache": str(tmp / f"{name}_mpk.npz"), "split": str(tmp / f"{name}_split.json"), "fold": "fold1",
            "has_test": True, "partial_split_ok": False, "disjoint": [["train", "val"], ["train", "test"], ["val", "test"]]}
dS, dB = make("slovo", 6, 10, [f"S{k}" for k in range(5)]), make("bukva", 4, 10, [f"B{k}" for k in range(5)])
AUG = dict(boundary_shift=5, boundary_inward_only=True, min_keep_frac=0.5, tempo=[0.5, 2.0], p_drop=0.1, p_dup=0.3,
           p_hand_drop=[0.2, 0.25], rot_deg=15.0, scale=[0.9, 1.1], shift=0.05, hand_jitter=0.002, window="mpk", spatial="mpk")
def cfg(data, steps, every, ft):   # общий рецепт (common.yaml 7563b5e) + модель Д-Т, как swap_model (T = 32)
    return {"model": {"arch": "sohn_shared", "dim": 192, "emb_dim": 256, "emb_scale": 16.0, "T": 32}, "data": data,
            "train": dict(optimizer="adamw", refit_trainval=False, ema=True, augment=True, select="best_val", merge_val=False,
                          batch_size=16, steps=steps, warmup_frac=0.05, lr=1e-3, weight_decay=0.05, eval_every=every,
                          ema_decay=0.999, ema_decay_short=0.99, ema_short_below=3000, ema_warmup=True),
            "loss": dict(label_smoothing=0.1, lambda_emb=0.5, beta_part=0.1), "finetune": ft,
            "synth": {"enabled": False}, "dominant": {"mode": "none"}, "aug": AUG}
FT0 = dict(init_from=False, freeze_steps=0, unfreeze_every=0, backbone_lr_mult=1.0)
cf = tmp / "g3_code.txt"
torch.set_num_threads(4)
pre_set = lambda d: {**d, "split": d["split"].replace("_split.json", "_split_pre.json"), "has_test": False, "disjoint": [["train", "val"]]}
# shortcut: S+B (две головы) не проверяю — _Base.forward в моей копии 7563b5e не принимает segs (у к2 на f6a5a72 принимает);
# предобучение — один набор (Slovo без теста), дальше путь тот же: init_from → заморозка → разморозка
pre = P.run(cfg(pre_set(dS), 24, 8, FT0), 0, tmp / "pre", None, 0, cf)
print("1) предобучение (Slovo, без теста): шагов", pre["steps_done"], "| доли пакета", pre["batch_share"], "| best_step", pre["best_step"],
      "| EMA decay", pre["ema_decay"], "| файлы:", sorted(p.name for p in (tmp / "pre").iterdir()))
m0 = P.SohnShared({"arch": "sohn_shared", "dim": 192, "emb_dim": 256, "emb_scale": 16.0, "T": 32}, {"slovo": 6})
FT = dict(init_from=str(tmp / "pre/best.pt"), freeze_steps=6, unfreeze_every=4, backbone_lr_mult=0.1)
pg, groups = param_groups(m0, FT, 1e-3)   # покрытие: assert внутри param_groups
name = {id(p): n for n, p in m0.named_parameters()}
gnames = [{name[id(p)] for mod in g for p in mod.parameters()} for g in groups]
heads = {name[id(p)] for p in pg[0]["params"]}
print("2) param_groups на дереве рычагов: групп кодировщика", len(groups), "| параметров в группах", [len(g) for g in gnames],
      "| голов", len(heads), "| головы:", sorted({n.split(".")[0] for n in heads}))
gbuf = [{f"{n}.{b}" for mod in g for n, mm in m0.named_modules() for b in ("running_mean", "running_var")
         if isinstance(mm, sohn.MaskedBN) and any(mm is x for x in mod.modules())} for g in groups]
sd0 = torch.load(tmp / "pre/best.pt", map_location="cpu")["model"]
for k, (ms, label) in enumerate(((6, "все 5 групп заморожены (шаги 1–6)"), (14, "шаг 14: группы 0–1 разморожены (с 7 и 11), 2–4 — нет"))):
    c = cfg(dS, 40, 7, FT); c["train"]["select"] = "last"   # веса последнего шага (EMA): сравнение не зависит от выбора best_val
    r = P.run(c, 1, tmp / f"ft{k}", ms, 0, cf)
    sd = torch.load(tmp / f"ft{k}/best.pt", map_location="cpu")["model"]
    same = [all(torch.equal(sd[n], sd0[n]) for n in gn | gb) for gn, gb in zip(gnames, gbuf)]
    lrs = r["trace"][-1][1]
    print(f"3.{k + 1}) дообучение, max_steps {ms} ({label}): веса и MaskedBN групп = предобучению: {same} | "
          f"замороженные на последнем шаге: {r['trace'][-1][2]} | lr кодировщика / lr голов = {lrs[1] / lrs[0]:.3f} | "
          f"emb изменилось: {not torch.equal(sd['emb.weight'], sd0['emb.weight'])} | best_step {r['best_step']}")
print("4) g3_code.txt:", json.loads(cf.read_text()))
