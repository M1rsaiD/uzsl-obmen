"""M13-Sohn: один прогон рецепта Sohn (reports/M13_sohn_prereg.md) на нашем протоколе.

cd osnovnaya_model && PYTHONPATH=src:../uzsign_e1/src python scripts/m13_train.py --set bukva --fold fold1 --seed 42
  --set slovo|bukva       протокол П (splits/<набор>_p5.json): режим fold — обучение на train, их ModelCheckpoint
                          (лучший val_loss по нашей валидации каждую эпоху) → тест лучшей; последняя — справочно
  --set sb                предобучение Slovo + Bukva (splits/{slovo,bukva}_pretrain_all.json, как у A1/SL-GCN корпуса):
                          режим fold, val = их val
  --set asl               Э0: проверка порта на asl-signs (их fold 0, ASL_FOLD0), --stop-epoch 83 при расписании 300 эпох
  --set uzsl              корпус LOSO (../uzsign_ai_v2/splits/uzsl_loso.json, фолды A1/SL-GCN): режим all (train + val,
                          последняя эпоха, как их обучение на всех данных); --init runs/M13/sb/all/seed<k>/best.pt —
                          вариант с предобучением (классификатор новый, остальное — веса предобучения)
Выход runs/UzSohn/<тег>/<фолд>/seed<k>/: predictions.csv (главный), predictions_{best,last}.csv, logits.npz, best.pt,
last.pt, log.csv, metrics.json, probs.npz (ids, probs float16, classes). Есть metrics.json — пропуск; оборванный каталог переносится в <тег>/_oborvan/.
Только проверка кода: --limit-train N --epochs E (выход в другой каталог, --out).
UzSohn (reports/UZSOHN_prereg.md): --variant 3,2,1 — идеи (3 поза + Н2-кисть, кэш cache/UZSOHN/<набор>_pose.npz; 2 граф
кисти; 1 вход по частям), по умолчанию — прежний Sohn; --inner-val (alluz, alluz_x10) — отбор: обучение на фолде без
внутреннего валидационного исполнителя V_k (и без его синтетики ×10), счёт — на V_k (тест S_k не читается).
UzSohn этап 1 (reports/UZSOHN_prereg.md): --input u02s,u02c,u01a,u04 — вход (sohn.preprocess_u; только --set sb и alluz, без
--variant); --save-from-epoch 200 — веса ep200…ep290.pt для U30 (только диск); отбор и его предобучения — затравка 48
(UZSOHN_SEL_SEED).
U33 (../uzsign_ai_v2/reports/U33_prereg.md, только --set alluz LOSO с --init): --arch a1 --recipe sohn — A1 на конвейере и
рецепте Sohn; --arch sohn --recipe a1 — Sohn с рецептом дообучения A1; по умолчанию (sohn, sohn) — прежний Sohn. --max-steps —
проба; --plan-only — план прогона для селфтеста (uzsign_ai_v2/scripts/u33.py selftest).
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mpk import sohn

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import m13_build  # noqa: E402

SPLIT = {"slovo": ROOT / "splits/slovo_p5.json", "bukva": ROOT / "splits/bukva_p5.json",
         "uzsl": UZ / "uzsign_ai_v2/splits/uzsl_loso.json",
         "alluz": UZ / "uzsign_ai_v2/splits/uzsl_allz_loso.json", "alluz_all": UZ / "uzsign_ai_v2/splits/uzsl_allz_all.json",  # ALL-UZ (ALLUZ_prereg.md)
         "alluz_x10": UZ / "uzsign_ai_v2/splits/uzsl_allz_x10_loso.json", "alluz_x10_all": UZ / "uzsign_ai_v2/splits/uzsl_allz_x10_all.json"}
X10_NPZ = UZ / "uzsign_ai_v2/datasets_local/UZSL_X10/uzsl_x10_sohn.npz"  # ALL-UZ ×10 (ALLUZ_X10_prereg.md): кадры уже без пустых
X10_SYN = "__x10p"
# Урок 23: в Python 3.14 метод запуска по умолчанию — forkserver: набор (AugSet) пересылается pickle КАЖДОМУ воркеру
# (asl-signs ≈ 3,4 ГБ × 6 + 4 воркера оценки → OOM при 40 ГБ). fork — общие страницы (копирование при записи), набор не пишется.
FORK = __import__("multiprocessing").get_context("fork")
PARTS = 8                                  # их 8 ядер TPU: пакет 512 = 8 частей по 64
# Э0 (поправка M13, 0059/0060): их сохранённый прогон — fold 0 tfrecords «5fold» (train 70 611 / valid 23 866). Фолды по
# исполнителям (23 866 = сумма 5 исполнителей, при случайном делении ≈ 18 895); точно 23 866 дают 5 наборов исполнителей —
# взят лексикографически первый (выбор до прогона, угроза в предрегистрации).
ASL_FOLD0 = {"2044", "16069", "22343", "28656", "49445"}
POSE_NPZ = {"alluz": ROOT / "cache/UZSOHN/alluz_pose.npz", "alluz_x10": ROOT / "cache/UZSOHN/alluz_x10_pose.npz",  # uzsohn_pose.py
            "slovo": ROOT / "cache/UZSOHN/slovo_pose.npz", "bukva": ROOT / "cache/UZSOHN/bukva_pose.npz"}
ASPECT_NPZ = {"alluz": UZ / "uzsign_ai_v2/cache/uzsl_allz.npz", "slovo": ROOT / "cache/slovo.npz", "bukva": ROOT / "cache/bukva.npz"}  # H/W, кэши МПК
CODES = UZ / "uzsign_ai_v2/data/signer_codes.local.json"   # только ПК
UZSOHN_SEL_SEED = 48   # UzSohn этап 1: отбор целиком (контроль sb/all/seed48, предобучения плеч, дообучение) — вне 42–47 подтверждения (к1 0123 п. 20)
IDEA = {1: "parts", 2: "hand_gcn", 3: "pose", 4: "hand_cross", 5: "ms_conv", 6: "attn_pool", 7: "cos_head", 8: "aux_shape"}  # UZSOHN_prereg


def inner_val(fold: str, folds: list[str]) -> str:
    """UZSOHN_prereg разд. 2: V_k — следующий по кругу исполнитель в порядке S11, S12, S01 … S09 (10 тестовых)."""
    code = json.loads(CODES.read_text())["signers"]
    order = sorted(folds, key=lambda f: (code[f] not in ("S11", "S12"), code[f]))
    assert len(order) == 10 and [code[f] for f in order[:3]] == ["S11", "S12", "S01"], [code[f] for f in order]
    v = order[(order.index(fold) + 1) % len(order)]
    assert v != fold
    return v


class Recs:
    """Записи после filter_nans в ОДНОМ непрерывном массиве (xy, absent) + смещения; recs[i] → (x, absent) — срезы.
    Урок 23: список из 94 тыс. мелких массивов лежал в куче вперемешку с объектами Python — счётчики ссылок и gc в каждом
    воркере DataLoader (fork) копировали страницы кучи → копия набора на воркер (OOM при 40 ГБ). Крупный массив — отдельная
    область mmap, воркеры её только читают. Содержимое recs[i] побайтно то же, что sohn.filter_nans(запись)."""

    def __init__(self, xy: np.ndarray, absent: np.ndarray, offsets: np.ndarray, extra: np.ndarray | None = None,
                 aspect: np.ndarray | None = None):
        keep = ~np.isnan(xy).all((1, 2))
        n = len(offsets) - 1
        cnt = np.add.reduceat(keep.astype(np.int64), offsets[:-1]) if n else np.zeros(0, np.int64)
        cnt[np.diff(offsets) == 0] = 0
        assert (cnt > 0).all(), f"записей без единой точки: {(cnt == 0).sum()} (filter_nans дал бы 1 пустой кадр)"
        # ×10: кэш уже без пустых кадров — без копии (≈ 7 ГБ; 3 прогона на ОЗУ 40 ГБ)
        self.xy, self.absent = (xy, absent) if keep.all() else (xy[keep], absent[keep])
        self.off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
        # UzSohn идея 3: точки позы [ΣT, 4, 2] кадр в кадр с xy (те же кадры убираются); recs[i] → 122 точки
        self.extra = None if extra is None else (extra if keep.all() else extra[keep])
        self.aspect = aspect   # UzSohn этап 1: H/W записи (u02s, u04); tokens ставит main (вход preprocess_u)

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        a, b = self.off[i], self.off[i + 1]
        if getattr(self, "extra", None) is not None:
            return np.concatenate([self.xy[a:b], self.extra[a:b]], 1), self.absent[a:b]
        return self.xy[a:b], self.absent[a:b]

    @staticmethod
    def concat(rs: list["Recs"]) -> "Recs":
        r = Recs.__new__(Recs)
        r.xy, r.absent = np.concatenate([x.xy for x in rs]), np.concatenate([x.absent for x in rs])
        r.off = np.concatenate([[0], np.cumsum(np.concatenate([np.diff(x.off) for x in rs]))]).astype(np.int64)
        ex = [getattr(x, "extra", None) for x in rs]
        assert all(e is None for e in ex) or all(e is not None for e in ex)
        r.extra = None if ex[0] is None else np.concatenate(ex)
        asp = [getattr(x, "aspect", None) for x in rs]
        r.aspect = None if asp[0] is None else np.concatenate(asp)
        return r


def aspect_of(set_: str, clips: list[dict]) -> np.ndarray:
    """H/W записей из кэша МПК (те же sample_id в том же порядке, что cache/M13/<набор>.npz)."""
    with np.load(ASPECT_NPZ[set_], allow_pickle=False) as z:
        assert [c["sample_id"] for c in json.loads(str(z["clips"]))] == [c["sample_id"] for c in clips], set_
        return z["aspect"].astype(np.float32)


def records(set_: str, fold: str, pose: bool = False, inp: bool = False):
    """→ recs [(x [T,118,2], absent [T,4])] после filter_nans, glosses, clips, (train, val, test) — индексы записей."""
    if set_ == "asl":
        fold = fold or "sohn0"
        c = m13_build.load("asl")
        o = c["offsets"]
        recs = Recs(c.pop("xy"), c.pop("absent"), o)
        if fold == "sohn0":
            vset = ASL_FOLD0
        else:  # поправка 2 (к1 0063): pf1–pf5 — 5 фолдов по исполнителям splits/asl_signs_cv5.json (группа k−1 = отложенный фолд)
            g = json.loads((ROOT / "splits/asl_signs_cv5.json").read_text())["groups"]
            vset = {p[3:] for p, k in g.items() if k == int(fold[2:]) - 1}
            assert len(vset) >= 4, vset
        va = [i for i, cl in enumerate(c["clips"]) if str(cl["signer_id"]) in vset]
        tr = [i for i, cl in enumerate(c["clips"]) if str(cl["signer_id"]) not in vset]
        assert len(tr) + len(va) == 94477 and (fold != "sohn0" or (len(tr), len(va)) == (70611, 23866)), (len(tr), len(va))
        return recs, [cl["gloss"] for cl in c["clips"]], c["clips"], (tr, va, [])
    if set_ == "sb":
        parts, glosses, clips, tr, va = [], [], [], [], []
        for ds in ("slovo", "bukva"):
            r, g, c, (a, b, _) = records(ds, None, pose, inp)
            off = sum(len(x) for x in parts)
            parts.append(r)
            glosses += [f"{ds}:{x}" for x in g]
            clips += c
            f = json.loads((ROOT / f"splits/{ds}_pretrain_all.json").read_text())["folds"]["all"]
            pos = {cl["sample_id"]: i for i, cl in enumerate(c)}
            tr += [off + pos[s] for s in f["train"]]
            va += [off + pos[s] for s in f["val"]]
        return Recs.concat(parts), glosses, clips, (tr, va, [])
    c = m13_build.load(str(X10_NPZ) if set_ == "alluz_x10" else set_)
    o = c["offsets"]
    ext = None
    if pose or inp:
        with np.load(POSE_NPZ[set_], allow_pickle=False) as z:
            assert np.array_equal(z["offsets"], o), "кадры позы ≠ кадры кэша Sohn"
            ext = z["xy"]
    recs = Recs(c.pop("xy"), c.pop("absent"), o, ext, aspect_of(set_, c["clips"]) if inp else None)
    glosses = [cl["gloss"] for cl in c["clips"]]
    if fold is None:
        return recs, glosses, c["clips"], (None, None, None)
    f = json.loads(SPLIT[f"{set_}_all" if set_ in ("alluz", "alluz_x10") and fold == "all" else set_].read_text())["folds"][fold]
    pos = {cl["sample_id"]: i for i, cl in enumerate(c["clips"])}
    return recs, glosses, c["clips"], tuple([pos[s] for s in f[k]] for k in ("train", "val", "test"))


class AugSet(Dataset):
    """Элемент (запись, позиция в потоке) → (признаки [T, 708], метка). Случайность аугментации — от (затравка, позиция).
    U33: a1aug — аугментации A1 вместо Sohn (a1_aug_sohn; aspect — H/W записи). UzSohn этап 1: u02 — общий с воркерами (fork)
    счётчик [примеров с u02*, из них без кадра с обоими плечами после масок — ветка Sohn, вариант (в) UZSOHN_prereg 4.1]."""

    def __init__(self, recs, labels, seed: int, aug: bool, a1aug: dict | None = None, aspect: np.ndarray | None = None):
        self.recs, self.labels, self.seed, self.aug = recs, labels, seed, aug
        self.a1aug, self.aspect = a1aug, aspect
        self.u02 = FORK.Array("q", 2) if aug else None   # оценка (predict) — без счётчика

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, item):
        i, pos = item
        x, ab = self.recs[i]
        tok = getattr(self.recs, "tokens", ())
        if self.aug and self.a1aug is not None:
            x = a1_aug_sohn(x, float(self.aspect[i]), self.a1aug, np.random.default_rng([self.seed, pos]))
        elif self.aug:
            x = sohn.augment(x, ab, np.random.default_rng([self.seed, pos]))
        if tok:   # UzSohn этап 1: x — 122 точки (аугментация как у идеи 3: отражение 11↔12, 13↔14)
            if self.aug and {"u02s", "u02c"} & set(tok):
                with self.u02.get_lock():
                    self.u02[0] += 1
                    self.u02[1] += not shoulders_seen(x)
            return sohn.preprocess_u(x, float(self.recs.aspect[i]), tok), self.labels[i]
        return (sohn.preprocess(x) if x.shape[1] == sohn.NUM_NODES else sohn.preprocess_pose(x)), self.labels[i]


def shoulders_seen(x: np.ndarray) -> bool:
    """Есть кадр с обоими плечами (точки 118, 119) — иначе опора U02 — ветка Sohn (sohn.center_scale_u)."""
    return bool(np.isfinite(x[:, sohn.NUM_NODES:sohn.NUM_NODES + 2]).all((1, 2)).any())


class Stream:
    """Их tf.data: repeat() → shuffle(32768) → padded_batch(512). Порядок эпохи — постоянная перестановка (их чтение
    TFRecord идёт в одном порядке файлов), буфер перемешивания 32768 — как ShuffleDataset."""

    def __init__(self, idx: list[int], steps: int, seed: int, bs: int, buf: int):
        self.idx, self.steps, self.seed, self.bs, self.buf = idx, steps, seed, bs, buf

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng = np.random.default_rng([self.seed, 7])
        base = rng.permutation(self.idx)
        gen = (int(base[k % len(base)]) for k in range(10 ** 12))
        buf = [next(gen) for _ in range(self.buf)]
        pos = 0
        for _ in range(self.steps):
            out = []
            for _ in range(self.bs):
                j = int(rng.integers(len(buf)))
                out.append((buf[j], pos))
                buf[j] = next(gen)
                pos += 1
            yield out


def collate_parts(items):
    n = len(items) // PARTS
    parts = []
    for p in range(PARTS):
        ch = items[p * n:(p + 1) * n]
        x, m = sohn.collate([a for a, _ in ch])
        parts.append((x, m, torch.tensor([b for _, b in ch])))
    return parts


def collate_eval(items):
    x, m = sohn.collate([a for a, _ in items])
    return x, m, torch.tensor([b for _, b in items])


@torch.no_grad()
def predict(model, recs, labels, idx, dev, ls: float) -> tuple[np.ndarray, float]:
    model.eval()
    ds = AugSet(recs, labels, 0, False)
    dl = DataLoader(ds, batch_size=256, collate_fn=collate_eval, num_workers=4, multiprocessing_context=FORK,
                    sampler=[(i, 0) for i in idx])
    out, loss, n = [], 0.0, 0
    for x, m, y in dl:
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            z = model(x.to(dev), m.to(dev)).float()
        loss += float(F.cross_entropy(z, y.to(dev), label_smoothing=ls, reduction="sum"))
        n += len(y)
        out.append(z.cpu().numpy())
    return np.concatenate(out), loss / max(n, 1)


def write_preds(path: Path, set_: str, logits: np.ndarray, idx, clips, labels, names) -> float:
    top = np.argsort(-logits, 1)[:, :5]
    ok = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "sample_id", "signer_id", "gloss", "pred", "correct", "top5"])
        for r, i in enumerate(idx):
            c = int(top[r, 0] == labels[i])
            ok += c
            w.writerow([set_, clips[i]["sample_id"], clips[i]["signer_id"], names[labels[i]], names[top[r, 0]], c,
                        "|".join(names[k] for k in top[r])])
    return ok / len(idx)


# ---------------------------------------------------------------- U33 «рецепт или архитектура» (uzsign_ai_v2/reports/U33_prereg.md)
# --arch a1 --recipe sohn: модель A1 (mpk.model, 85 точек, окно 32) на конвейере и рецепте Sohn; --arch sohn --recipe a1: Sohn
# с рецептом дообучения A1 (mpk.train: AdamW, прогрев + косинус, заморозка и послойная разморозка, EMA, аугментации A1).
A1_CACHE = UZ / "uzsign_ai_v2/cache/uzsl_allz.npz"   # 85 точек ALL-UZ (те же записи, что cache/M13/alluz.npz)
LATE_P = 0.8                                          # LateDropout(0.8) Sohn [c16]


def a1_cfg(fold: str, n: int, seed: int) -> dict:
    """Конфиг клетки A1 · A1 этого фолда ровно как alluz._a1_run (loso.cfg_for): модель и рецепт A1 (train, finetune, aug, loss)."""
    sys.path.insert(0, str(UZ / "uzsign_ai_v2/scripts"))
    from loso import PRETRAIN, cfg_for
    return cfg_for(SPLIT["alluz"], fold, n, True, PRETRAIN.format(seed=seed), A1_CACHE, True, "_allz", ema_warmup=False)


class A1Recs:
    """Записи кэша 85 точек — срезы общих массивов (fork: страницы общие); recs[i] → (xy, vis, present, H/W)."""

    def __init__(self, c: dict):
        assert (c["lo"] == 0).all() and (c["hi"] == np.diff(c["offsets"])).all(), "ALL-UZ: отрезок = запись"
        self.xy, self.vis, self.pres, self.off, self.asp = c["xy"], c["vis"], c["present"], c["offsets"], c["aspect"]

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        a, b = self.off[i], self.off[i + 1]
        return self.xy[a:b].astype(np.float32), self.vis[a:b].astype(np.float32), self.pres[a:b], float(self.asp[i])


def records_a1(fold: str):
    from mpk.data import load
    c = load(A1_CACHE)
    pos = {cl["sample_id"]: i for i, cl in enumerate(c["clips"])}
    f = json.loads(SPLIT["alluz"].read_text())["folds"][fold]
    return A1Recs(c), [cl["gloss"] for cl in c["clips"]], c["clips"], tuple([pos[s] for s in f[k]] for k in ("train", "val", "test"))


def a1_window(L: int, T: int) -> np.ndarray:
    """Окно проверки A1 (mpk.augment.window, aug=None): T равномерных кадров записи длины L."""
    return np.minimum(np.floor((np.arange(T) + 0.5) * L / T).astype(np.int64), L - 1)


def a1_input(xy, c, pres, T: int):
    """= mpk.augment.batch(…, aug=None) одной записи: x [T,85,3] (x, y, c), m [T,4] (селфтест u33.py — бит в бит)."""
    i = a1_window(len(xy), T)
    return np.concatenate([xy[i], c[i, :, None]], -1).astype(np.float32), pres[i].astype(np.float32)


@__import__("functools").cache
def flip85() -> np.ndarray:
    """Перестановка сторон 85 точек A1 (mpk.data.mirror_points); 4 точки губ без пары среди 85 — на месте."""
    from mpk.data import mirror_points
    perm = mirror_points()[:85]
    return np.where(perm < 85, perm, np.arange(85))  # shortcut: пар 4 точек нет в кэше ALL-UZ (U33_prereg разд. 3 п. 1)


def a1_sohn_aug(xy, vis, pres, rng, T: int):
    """sohn.augment на 85 точках A1 (U33_prereg разд. 3 п. 1) → xy, c [T,85], present [T,4] на T кадрах окна A1: точка есть =
    конечна, координаты отсутствующих — линейная интерполяция по времени (как сборка кэша), c позы — видимость."""
    from mpk.data import PART_SLICES, POINT_PART
    x = np.where(pres[:, POINT_PART, None], xy, np.nan).astype(np.float32)
    v = np.where(pres[:, :1], vis, np.nan).astype(np.float32)
    x, v = sohn.augment(x, None, rng, flip=flip85(), extra=v, extra_flip=[1, 0, 3, 2, 5, 4])
    ok = np.isfinite(x).all(-1)
    t = a1_window(len(x), T)
    out = np.zeros((T, 85, 2), np.float32)
    for j in np.flatnonzero(ok.any(0)):
        s = np.flatnonzero(ok[:, j])
        for k in (0, 1):
            out[:, j, k] = np.interp(t, s, x[s, j, k])
    c = ok[t].astype(np.float32)
    c[:, :6] *= np.clip(np.nan_to_num(v[t]), 0, 1)   # видимость — вероятность; бикубика даёт выброс до 1 + 6e-5
    return out, c, np.stack([ok[t][:, s].any(1) for s in PART_SLICES], 1)


def a1_aug_sohn(x, asp: float, aug: dict, rng):
    """Аугментации A1 на 118 точках Sohn (U33_prereg разд. 3 п. 2), NaN — нет точки. Время — окно A1 (mpk.augment.window) с
    T = длине записи; пространство — формулы mpk.augment.batch: выпадение кистей по кадрам (→ NaN), поворот, масштаб и сдвиг в
    изотропных координатах (y · H/W), дрожание точек кистей."""
    from mpk.augment import window
    assert aug["window"] == "mpk" and aug["spatial"] == "mpk", aug
    L = len(x)
    one = lambda v: torch.tensor([v])
    x = x[window(one(0), one(L), one(L), L, aug, torch.Generator().manual_seed(int(rng.integers(2 ** 62))))[0].numpy()].copy()
    lo_p, hi_p = aug["p_hand_drop"]
    drop = rng.random((L, 2)) < lo_p + rng.random() * (hi_p - lo_p)
    m = np.zeros(x.shape[:2], bool)
    m[:, sohn.LH_LOC], m[:, sohn.RH_LOC] = drop[:, :1], drop[:, 1:]
    x[m] = np.nan
    ang = (rng.random() * 2 - 1) * np.radians(aug["rot_deg"])
    sc = aug["scale"][0] + rng.random() * (aug["scale"][1] - aug["scale"][0])
    cs, sn = np.cos(ang) * sc, np.sin(ang) * sc
    px, py = x[..., 0] - 0.5, (x[..., 1] - 0.5) * asp
    sh = (rng.random(2) * 2 - 1) * aug["shift"]
    px, py = cs * px - sn * py + sh[0], sn * px + cs * py + sh[1]
    h = sohn.LH_LOC + sohn.RH_LOC
    jit = rng.standard_normal((L, len(h), 2)) * aug["hand_jitter"]
    px[:, h] += jit[..., 0]
    py[:, h] += jit[..., 1]
    return np.stack([px + 0.5, py / asp + 0.5], -1).astype(np.float32)


class A1Set(Dataset):
    """Запись кэша 85 точек → (x [T,85,3], m [T,4], H/W, метка) — вход A1; aug — аугментации Sohn до окна (a1_sohn_aug)."""

    def __init__(self, recs: A1Recs, labels, seed: int, aug: bool, T: int):
        self.recs, self.labels, self.seed, self.aug, self.T = recs, labels, seed, aug, T

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, item):
        from mpk.data import POINT_PART
        i, pos = item
        xy, vis, pres, asp = self.recs[i]
        if self.aug:
            xy, c, pres = a1_sohn_aug(xy, vis, pres, np.random.default_rng([self.seed, pos]), self.T)
        else:
            c = pres[:, POINT_PART].astype(np.float32)
            c[:, :6] = vis * pres[:, :1]
        x, m = a1_input(xy, c, pres, self.T)
        return x, m, asp, self.labels[i]


def collate_a1(items):
    return (torch.from_numpy(np.stack([a[0] for a in items])), torch.from_numpy(np.stack([a[1] for a in items])),
            torch.tensor([a[2] for a in items], dtype=torch.float32), torch.tensor([a[3] for a in items]))


def collate_a1_parts(items):
    n = len(items) // PARTS
    return [collate_a1(items[p * n:(p + 1) * n]) for p in range(PARTS)]


class A1Net(torch.nn.Module):
    """A1 (mpk.model, конфиг A1 · A1) с late dropout рецепта Sohn на z перед основной головой и головой вложения."""

    def __init__(self, mc: dict, n_classes: int):
        super().__init__()
        from mpk.model import build_model
        self.m = build_model(mc, {"uzsl": n_classes})
        self.late_dropout_on = False

    def forward(self, x, m, a):
        z, tok, pm, _ = self.m.encode(x, m, a)
        if self.training and self.late_dropout_on:
            z = F.dropout(z, LATE_P, True)
        return self.m._heads(z, tok, pm, "uzsl")


def a1_loss_fn(model: A1Net, lc: dict):
    """Потеря A1 (mpk.train.seg_loss): CE + λ_emb · CE(вложение) + β_part · Σ CE(части); головы — архитектура A1."""
    ce = lambda lg, y: F.cross_entropy(lg.float(), y, label_smoothing=lc["label_smoothing"])

    def f(x, m, a, y):
        r = model(x, m, a)
        loss = ce(r["logits"], y)
        if lc["lambda_emb"]:
            loss = loss + lc["lambda_emb"] * ce(r["emb_logits"], y)
        if lc["beta_part"] and r["part_logits"]:
            loss = loss + lc["beta_part"] * sum(ce(p, y) for p in r["part_logits"])
        return loss
    return f


@torch.no_grad()
def predict_a1(model: A1Net, recs, labels, idx, dev, T: int) -> np.ndarray:
    model.eval()
    dl = DataLoader(A1Set(recs, labels, 0, False, T), batch_size=256, collate_fn=collate_a1, num_workers=4,
                    multiprocessing_context=FORK, sampler=[(i, 0) for i in idx])
    out = []
    for x, m, a, _ in dl:
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            out.append(model(x.to(dev), m.to(dev), a.to(dev))["logits"].float().cpu().numpy())
    return np.concatenate(out)


class EpochStream:
    """Порядок рецепта A1 (mpk.train): подряд случайные перестановки обучающих записей, пакет bs (через границу эпохи)."""

    def __init__(self, idx: list[int], steps: int, seed: int, bs: int):
        self.idx, self.steps, self.seed, self.bs = idx, steps, seed, bs

    def __len__(self):
        return self.steps

    def __iter__(self):
        rng, order, pos = np.random.default_rng([self.seed, 7]), np.empty(0, np.int64), 0
        for _ in range(self.steps):
            if len(order) < self.bs:
                order = np.concatenate([order, rng.permutation(self.idx)])
            b, order = order[:self.bs], order[self.bs:]
            yield [(int(i), pos + k) for k, i in enumerate(b)]
            pos += self.bs


def sohn_groups(model) -> list[list]:
    """Группы кодировщика Sohn сверху вниз для послойной разморозки (как MPK.layer_groups): верх + последний блок, блоки по
    одному, вход (stem + BN); голова (classifier) сюда не входит."""
    b = list(model.blocks)
    return [[model.top, b[-1]], *([x] for x in reversed(b[:-1])), [model.stem, model.stem_bn]]


def train_recipe_a1(model, ds, train: list[int], seed: int, c: dict, run_steps: int, dev, workers: int, lw, log) -> torch.nn.Module:
    """Рецепт дообучения A1 (mpk.train.run: AdamW, прогрев + косинус, голова lr / кодировщик × 0,1, заморозка и послойная
    разморозка с BN замороженных групп в режиме вывода, EMA весов и буферов) для SohnNet, пакет 128 одним проходом → EMA-модель."""
    from mpk.train import ema_avg_fn
    from uzsign_e1.train import lr_factor
    tc, ft, ls = c["train"], c["finetune"], c["loss"]["label_smoothing"]
    steps = tc["steps"]
    groups = sohn_groups(model)
    gp = [[p for mod in g for p in mod.parameters()] for g in groups]
    heads = list(model.classifier.parameters())
    ids = [id(p) for g in gp for p in g] + [id(p) for p in heads]
    assert len(ids) == len(set(ids)) == len(list(model.parameters())), "группы не покрывают модель"
    opt = torch.optim.AdamW([{"params": heads, "lr": tc["lr"]}] + [{"params": g, "lr": tc["lr"] * ft["backbone_lr_mult"]} for g in gp],
                            lr=tc["lr"], weight_decay=tc["weight_decay"])
    warm = max(1, round(tc["warmup_frac"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, warm, steps))
    decay = tc["ema_decay"] if steps >= tc["ema_short_below"] else tc["ema_decay_short"]
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=ema_avg_fn(decay, tc["ema_warmup"]), use_buffers=True)
    dl = DataLoader(ds, batch_sampler=EpochStream(train, run_steps, seed, tc["batch_size"]), collate_fn=collate_eval,
                    num_workers=workers, prefetch_factor=2, multiprocessing_context=FORK, persistent_workers=True)
    every, run_loss, t0 = max(1, tc["eval_every"]), 0.0, time.time()
    for step, (x, m, y) in enumerate(dl, 1):
        x, m, y = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        model.train()
        fz = [k for k in range(len(groups)) if step <= ft["freeze_steps"] + k * ft["unfreeze_every"]]
        for k in fz:
            for mod in groups[k]:
                for b in mod.modules():
                    if isinstance(b, sohn.MaskedBN):
                        b.eval()
        model.set_pass(0)
        with torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
            loss = F.cross_entropy(model(x, m).float(), y, label_smoothing=ls)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        for k in fz:
            for p in opt.param_groups[1 + k]["params"]:
                p.grad = None   # AdamW пропускает параметры без градиента (и затухание тоже), как mpk.train
        lr = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
        model.commit_bn()
        ema.update_parameters(model)
        run_loss += float(loss)
        if step % every == 0 or step == run_steps:
            lw.writerow(["", step, run_loss / (every if step % every == 0 else step % every), "", "", lr, round(time.time() - t0)])
            log.flush()
            run_loss = 0.0
    return ema.module


def uzsohn_code() -> tuple[list[str], list[str]]:
    """Код этапа 1 UzSohn (scripts/uzsohn_code.txt; UZSOHN_prereg разд. 14): пути osnovnaya_model и uzsign_ai_v2 («v2:»)."""
    ls = [s.split("#")[0].strip() for s in Path(__file__).with_name("uzsohn_code.txt").read_text().splitlines()]
    ls = [s for s in ls if s]
    return [s for s in ls if not s.startswith("v2:")], [s[3:] for s in ls if s.startswith("v2:")]


def git(repo: Path, paths=("src/mpk/sohn.py", "scripts")) -> dict:
    run = lambda *a: subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", *paths))}


def main(argv: list[str] | None = None) -> dict | None:
    # перерыв на выходные (0064/0065): новый прогон ждёт Пн 08:00; идущие не трогаются (U33 --plan-only — без обучения)
    import subprocess
    if "--plan-only" not in (sys.argv if argv is None else argv):
        subprocess.run(["bash", str(Path(__file__).with_name("weekend_gate.sh"))], check=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["slovo", "bukva", "uzsl", "sb", "asl", "alluz", "alluz_x10"])
    ap.add_argument("--fold", default="all")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--init", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=sohn.CFG["epoch"])
    ap.add_argument("--stop-epoch", type=int, default=0, help="остановка после эпохи N при расписании на --epochs (Э0: 83)")
    ap.add_argument("--variant", default="", help="UzSohn: идеи через запятую из 1…8 (пусто — Sohn; 8 — заготовка)")
    ap.add_argument("--inner-val", action="store_true", help="UzSohn отбор: счёт на V_k, обучение без V_k")
    ap.add_argument("--input", default="", help="UzSohn этап 1: токены входа через запятую из u02s, u02c, u01a, u04 (пусто — Sohn)")
    ap.add_argument("--save-from-epoch", type=int, default=0, help="U30 (UZSOHN_prereg): веса ep<N>.pt каждые 10 эпох с N (0 — нет)")
    ap.add_argument("--arch", default="sohn", choices=["sohn", "a1"], help="U33: a1 — модель A1 (85 точек) на конвейере Sohn")
    ap.add_argument("--recipe", default="sohn", choices=["sohn", "a1"], help="U33: a1 — рецепт дообучения A1")
    ap.add_argument("--max-steps", type=int, default=0, help="U33 проба: обрыв после N шагов (расписание не меняется)")
    ap.add_argument("--plan-only", action="store_true", help="U33 селфтест: план прогона (dict), без обучения")
    a = ap.parse_args(argv)
    ideas = {int(v) for v in a.variant.split(",") if v}
    tokens = tuple(t for t in sohn.INPUT_TOKENS if t in a.input.split(","))
    assert set(a.input.split(",")) - {""} == set(tokens), a.input
    assert not (tokens and ideas), "этап 1 (--input) и идеи 0107 (--variant) не смешиваются"
    assert not tokens or a.set in ("sb", "alluz"), "этап 1: предобучение sb и отбор alluz (база — Sohn ALL-UZ)"
    assert ideas <= set(IDEA), ideas
    flags = {IDEA[k]: k in ideas for k in sorted(IDEA)}
    assert not a.inner_val or (a.set in ("alluz", "alluz_x10") and a.fold != "all")
    u33 = (a.arch, a.recipe) != ("sohn", "sohn")
    assert not u33 or (a.arch != a.recipe and a.set == "alluz" and a.fold != "all" and not ideas and not a.inner_val and a.init), \
        "U33: перекрёстные клетки ALL-UZ LOSO с --init (uzsign_ai_v2/reports/U33_prereg.md)"
    assert not (tokens and u33), "этап 1 (--input) и U33 не смешиваются"
    if a.init and a.arch == "sohn":   # к1 0120 п. 6: у плеча своё предобучение S+B — вход весов = вход прогона
        im = Path(a.init).with_name("metrics.json")
        # отбор (--inner-val) — и у контроля: его sb/all/seed48 тоже своё, оборванное без metrics.json не годится (р4)
        assert im.is_file() or not (tokens or a.inner_val), f"{im}: нет metrics.json предобучения (не завершено?)"
        if im.is_file():
            m0 = json.loads(im.read_text())
            assert (m0.get("input") or []) == list(tokens), f"--init {a.init}: вход весов {m0.get('input')} ≠ {list(tokens)}"
            # бюджет предобучения — до дообучения, те же условия, что uzsohn_sel.full_budget: цепочка пропускает готовое по
            # metrics.json, а report сверяет его лишь в конце этапа (раунд 4 проверки р4)
            assert not (tokens or a.inner_val) or (m0.get("steps_run") == m0.get("steps") and m0.get("limit_train") == 0 and
                    m0.get("epochs") == sohn.CFG["epoch"] and not m0.get("stop_epoch") and not m0.get("max_steps")), \
                f"--init {a.init}: урезанное предобучение " + str({k: m0.get(k) for k in ("steps_run", "steps", "limit_train", "epochs",
                                                                                     "stop_epoch", "max_steps")}) + \
                " — восстановление: каталог этого предобучения в копию, затем перезапуск юнита (цепочка пропускает предобучение " \
                "с metrics.json — простой перезапуск упадёт снова; прогоны отбора от него не стартуют; UZSOHN_prereg разд. 4.4)"
    cfg = sohn.CFG
    tag = a.tag or a.set
    out = Path(a.out) if a.out else ROOT / "runs/UzSohn" / tag / a.fold / f"seed{a.seed}"
    if (out / "metrics.json").is_file():
        print(f"{out}: есть")
        return
    # замок на прогон (ALL-UZ, добиратель uz-allz-sb): O_EXCL + проверка процессов — один прогон не идёт дважды,
    # идущий в другом процессе каталог не переносится в _oborvan
    import atexit
    lock = out.parent / f"seed{a.seed}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        pid, _, bid = (lock.read_text() or "0").strip().partition(" ")
        pid = int(pid or 0)
        try:   # PID прошлой загрузки достаётся чужим процессам: живой замок — та же загрузка (старый замок — без неё) и m13_train
            alive = pid > 0 and bid in ("", boot) and b"m13_train.py" in Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            alive = False
        if alive:
            print(f"{out}: занят (замок, pid {pid})")
            return
        lock.rename(lock.with_name(lock.name + f".stale_{time.strftime('%Y%m%d_%H%M%S')}"))
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, f"{os.getpid()} {boot}".encode())
    os.close(fd)
    atexit.register(lambda: lock.exists() and lock.rename(lock.with_name(lock.name + f".done_{os.getpid()}")))
    pat = f"m13_train.py --set {a.set} --fold {a.fold} --seed {a.seed} --tag {tag}( |$)"
    others = [int(p) for p in subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True).stdout.split()
              if int(p) not in (os.getpid(), os.getppid())]
    if any(open(f"/proc/{p}/stat").read().split()[3] != str(os.getpid()) for p in others if Path(f"/proc/{p}").exists()):
        print(f"{out}: занят (идёт в другом процессе)")
        return
    if out.exists():
        dst = out.parents[1] / "_oborvan" / f"{a.fold}_seed{a.seed}_{time.strftime('%Y%m%dT%H%M%S')}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.rename(dst)
        print(f"оборванный прогон перенесён → {dst}")
    out.mkdir(parents=True)
    # код — до обучения (он и загружен); у этапа 1 UzSohn — только его файлы (uzsohn_code.txt): посторонний скрипт не «dirty» (р3)
    code = git(ROOT, uzsohn_code()[0] if tokens or a.inner_val or a.seed == UZSOHN_SEL_SEED else ("src/mpk/sohn.py", "scripts"))
    t0 = time.time()
    recs, glosses, clips, (tr, va, te) = records_a1(a.fold) if a.arch == "a1" else \
        records(a.set, None if a.set == "sb" else a.fold, pose=flags["pose"], inp=bool(tokens))
    recs.tokens = tokens
    mode = "all" if a.set in ("uzsl", "alluz", "alluz_x10") else "fold"
    lab_idx = tr + va + te
    names = sorted({glosses[i] for i in lab_idx})                    # как train.prepare_set / build_data Г3
    lab = {g: k for k, g in enumerate(names)}
    labels = np.full(len(recs), -1, np.int64)
    labels[lab_idx] = [lab[glosses[i]] for i in lab_idx]
    train = tr + va if mode == "all" else tr
    vk = None
    if a.inner_val:   # голова — те же классы, что у фолда (names выше); тест S_k дальше не используется
        folds = [f for f in json.loads(SPLIT[a.set].read_text())["folds"] if f != "all"]
        vk = inner_val(a.fold, folds)
        import re
        who = lambda i: re.sub(r"^syn_x10p\d+_", "", clips[i]["signer_id"])   # синтетика — исполнитель исходной записи
        # счёт V_k — по правилу теста ALL-UZ: записи V_k из теста его фолда (без 5 «другой»), чей класс есть в обучении
        vt = set(json.loads(SPLIT[a.set].read_text())["folds"][vk]["test"])
        train = [i for i in train if who(i) != vk]
        cls_tr = {glosses[i] for i in train}
        te = [i for i in tr + va if clips[i]["sample_id"] in vt and glosses[i] in cls_tr]
        assert te and all(who(i) == vk for i in te) and not any(who(i) == a.fold for i in train + te), (vk, len(te))
    if a.limit_train:   # только проверка кода; без условия train[:0] обнулял каждый прогон отбора (9c4abfe)
        train = train[:a.limit_train]
    assert not set(train) & set(te), "тест в обучении"
    # ×10: «эпоха» и бюджет шагов — по РЕАЛЬНЫМ записям (= ALL-UZ), синтетика лишь расширяет пул выборки
    n_epoch = sum(X10_SYN not in clips[i]["sample_id"] for i in train) if a.set == "alluz_x10" else len(train)
    assert a.set != "alluz_x10" or not any(X10_SYN in clips[i]["sample_id"] for i in te), "синтетика в тесте"
    spe = n_epoch // cfg["batch_size"]
    assert spe >= 1, len(train)
    total = a.epochs * spe
    late = cfg["dropout_start_epoch"] * spe
    awp0 = cfg["awp_start_epoch"] * spe
    # U33: конфиг клетки A1 · A1 этого фолда — модель A1 и/или рецепт A1; бюджет рецепта A1 — его шаги и пакет
    a1c = a1_cfg(a.fold, len(train), a.seed if a.arch == "a1" else 0) if u33 else None
    bs = cfg["batch_size"]
    if a.recipe == "a1":
        total, bs, spe, late, awp0 = a1c["train"]["steps"], a1c["train"]["batch_size"], None, None, None
    plan = {"n_train": len(train), "n_classes": len(names), "steps": total, "steps_per_epoch": spe, "awp_start_step": awp0,
            "late_dropout_start_step": late, "batch_size": bs, "test_ids": sorted(clips[i]["sample_id"] for i in te),
            "cfg": cfg if a.recipe == "sohn" else {k: a1c[k] for k in ("train", "finetune", "aug", "loss")},
            "a1_model": a1c["model"] if a.arch == "a1" else None}
    if a.plan_only:
        return plan
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if a.arch == "a1":
        from mpk.train import load_init
        model = A1Net(a1c["model"], len(names)).to(dev)
        assert a.init == a1c["finetune"]["init_from"], (a.init, a1c["finetune"]["init_from"])   # предобучение A1 · A1 (U34)
        init_info = load_init(model.m, a.init)
        for b in model.modules():   # 8 частей: бегущие средние BN — на каждой части → импульс / 8 (U33_prereg разд. 3 п. 3)
            if isinstance(b, torch.nn.modules.batchnorm._BatchNorm):
                b.momentum /= PARTS
    else:
        model = sohn.SohnNet(len(names), cfg["dim"], channels=sohn.input_channels(tokens), **flags).to(dev)
        init_info = None
    if a.init and a.arch == "sohn":
        sd = torch.load(a.init, map_location=dev)["model"]
        keep = {k: v for k, v in sd.items() if not k.startswith("classifier.")}
        miss = model.load_state_dict(keep, strict=False)
        new = [k for k in miss.missing_keys if k.startswith(("stem_extra.", "gcn.", "part_mlp.", "cross.", "pool_score.", "cos_scale"))
               or ".dw_ms." in k]   # UzSohn: новые модули — как созданы
        assert set(miss.missing_keys) - set(new) == {"classifier.weight", "classifier.bias"} and not miss.unexpected_keys, miss
        init_info = {"init_from": a.init, "loaded": len(keep)}
    run_steps = a.stop_epoch * spe if a.stop_epoch else total
    if a.max_steps:
        run_steps = min(run_steps, a.max_steps)
    best, best_epoch = float("inf"), None
    log = open(out / "log.csv", "w", newline="")
    lw = csv.writer(log)
    lw.writerow(["epoch", "step", "loss", "val_loss", "val_top1", "lr", "sec"])
    run_loss, t1 = 0.0, time.time()
    if a.recipe == "a1":   # Sohn с рецептом A1: H/W записи — из кэша 85 точек (те же sample_id)
        with np.load(A1_CACHE, allow_pickle=False) as z:
            p85 = {cl["sample_id"]: i for i, cl in enumerate(json.loads(str(z["clips"])))}
            asp = np.array([z["aspect"][p85[cl["sample_id"]]] for cl in clips], np.float32)
        model = train_recipe_a1(model, AugSet(recs, labels, a.seed, True, a1c["aug"], asp), train, a.seed, a1c, run_steps, dev,
                                a.workers, lw, log)
    else:
        opt = sohn.RAdamLookahead(model.parameters())
        ds = AugSet(recs, labels, a.seed, True) if a.arch == "sohn" else A1Set(recs, labels, a.seed, True, a1c["model"]["T"])
        dl = DataLoader(ds, batch_sampler=Stream(train, run_steps, a.seed, cfg["batch_size"], cfg["shuffle"]),
                        collate_fn=collate_parts if a.arch == "sohn" else collate_a1_parts, num_workers=a.workers, prefetch_factor=2,
                        multiprocessing_context=FORK, persistent_workers=True)   # урок 23: пакет asl до ≈ 0,3 ГБ
        loss_fn = a1_loss_fn(model, a1c["loss"]) if a.arch == "a1" else None
        for step, parts in enumerate(dl):
            parts = [tuple(t.to(dev, non_blocking=True) for t in p) for p in parts]
            model.train()
            model.late_dropout_on = step >= late
            lr, wd = sohn.schedules(step, total, cfg)
            run_loss += sohn.train_step(model, opt, parts, lr, wd, awp_on=cfg["awp"] and step >= awp0,
                                        awp_lambda=cfg["awp_lambda"], ls=cfg["label_smoothing"], amp=dev.type == "cuda", loss_fn=loss_fn)
            if (step + 1) % spe == 0:
                ep = (step + 1) // spe
                vl, vt = "", ""
                if mode == "fold":
                    z, vloss = predict(model, recs, labels, va, dev, cfg["label_smoothing"])
                    vt = float((z.argmax(1) == labels[va]).mean())
                    vl = vloss
                    if vloss < best:                                        # ModelCheckpoint(save_best_only, mode='min')
                        best, best_epoch = vloss, ep
                        torch.save({"model": model.state_dict(), "epoch": ep, "names": names}, out / "best.pt")
                lw.writerow([ep, step + 1, run_loss / spe, vl, vt, lr, round(time.time() - t0)])
                log.flush()
                run_loss = 0.0
                if a.save_from_epoch and a.save_from_epoch <= ep < a.epochs and ep % 10 == 0:   # U30: только диск
                    torch.save({"model": model.state_dict(), "epoch": ep, "names": names}, out / f"ep{ep}.pt")
    log.close()
    train_sec = time.time() - t1
    torch.save({"model": model.state_dict(), "epoch": a.epochs, "names": names}, out / "last.pt")
    res = {}
    if te:
        if a.arch == "a1":
            z_last = predict_a1(model, recs, labels, te, dev, a1c["model"]["T"])
        else:
            z_last, _ = predict(model, recs, labels, te, dev, cfg["label_smoothing"])
        res["top1_last"] = write_preds(out / "predictions_last.csv", a.set, z_last, te, clips, labels, names)
        z_main = z_last
        if mode == "fold":
            model.load_state_dict(torch.load(out / "best.pt", map_location=dev)["model"])
            z_main, _ = predict(model, recs, labels, te, dev, cfg["label_smoothing"])
            res["top1_best"] = write_preds(out / "predictions_best.csv", a.set, z_main, te, clips, labels, names)
        res["top1"] = write_preds(out / "predictions.csv", a.set, z_main, te, clips, labels, names)
        # единый формат вероятностей для ансамблей Э3 (0059 п. 7): ids, probs float16 (softmax главной контрольной точки), classes
        pz = torch.from_numpy(z_main).softmax(-1).numpy().astype(np.float16)
        np.savez(out / "probs.npz", ids=np.array([clips[i]["sample_id"] for i in te]), probs=pz, classes=np.array(names))
        np.savez(out / "logits.npz", logits=z_main.astype(np.float32), sample_ids=np.array([clips[i]["sample_id"] for i in te]),
                 labels=labels[te], last=z_last.astype(np.float32))
    (out / "labels.json").write_text(json.dumps(names, ensure_ascii=False))
    u02n = None   # вариант (в): доля ветки Sohn при обучении и на оценке (тест V_k или val S+B) — в печать (к1 0124 п. 3, к3 0032 п. 3)
    if {"u02s", "u02c"} & set(tokens):
        ev = te or (va if mode == "fold" else [])
        u02n = {"train_examples": int(ds.u02[0]), "train_sohn_branch": int(ds.u02[1]), "eval_records": len(ev),
                "eval_sohn_branch": sum(not shoulders_seen(recs[i][0]) for i in ev)}
    m = {"set": a.set, "tag": tag, "fold": a.fold, "seed": a.seed, "mode": mode, "select": "best_val_loss" if mode == "fold" else "last",
         "best_epoch": best_epoch, "best_val_loss": best if best_epoch else None, **res,
         "n_train": len(train), "n_train_real": n_epoch, "n_val": len(va) if mode == "fold" else 0, "n_test": len(te), "n_classes": len(names), "stop_epoch": a.stop_epoch or None, "steps_run": run_steps,
         "steps_per_epoch": spe, "epochs": a.epochs, "steps": total, "awp_start_step": awp0, "late_dropout_start_step": late,
         "train_sec": train_sec, "steps_per_sec": run_steps / train_sec, "total_sec": time.time() - t0,
         "params_keras_count": sohn.keras_param_count(model) if a.arch == "sohn" else None, "device": torch.cuda.get_device_name() if dev.type == "cuda" else "cpu",
         "git_osnovnaya_model": code, "sohn_repo_commit": "1ed5528244812a6e3d8fe2ad6d8b629aa6546dcc",
         "tf_utils_commit": "85b8d5ff93996b1d6b14ab77e4dc60f960de103e", "cfg": cfg, "init": init_info,
         "limit_train": a.limit_train, "variant": sorted(ideas, reverse=True), "flags": flags, "inner_val": vk,
         "input": list(tokens), "u02_sohn_branch": u02n, "channels": sohn.input_channels(tokens),
         "save_from_epoch": a.save_from_epoch or None,
         "params_torch": sum(p.numel() for p in model.parameters()),
         "gpu_max_mem_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev.type == "cuda" else None}
    if u33:
        m |= {"arch": a.arch, "recipe": a.recipe, "batch_size": bs, "cfg": plan["cfg"], "a1_model": plan["a1_model"],
              "max_steps": a.max_steps or None, "late_dropout_p": LATE_P if a.recipe == "sohn" else None,
              "prereg": "uzsign_ai_v2/reports/U33_prereg.md"}
    (out / "metrics.json").write_text(json.dumps(m, indent=1, ensure_ascii=False))
    print(json.dumps({k: m[k] for k in ("tag", "fold", "seed", "top1", "best_epoch", "steps", "steps_per_sec") if k in m}, ensure_ascii=False))


if __name__ == "__main__":
    main()
