"""M13-Sohn: перенос TF → PyTorch 1:1 модели, входа и рецепта 1-го места Kaggle GISLR (Hoyeol Sohn, 2023).

Источник (MIT): github.com/hoyso48/Google---Isolated-Sign-Language-Recognition-1st-place-solution @ 1ed55282,
ноутбук ISLR_1st_place_Hoyeol_Sohn.ipynb (клон — third_party/sohn_islr); их tf-utils (MIT) @ 85b8d5ff — расписание
OneCycleLR и AWP; tensorflow-addons 0.20 — RectifiedAdam и Lookahead. Номера ячеек ноутбука — в комментариях [cN].
Предрегистрация и список расхождений — reports/M13_sohn_prereg.md. Общий код mpk не меняется (только этот файл).

Запись данных (scripts/m13_build.py): xy [T, 118, 2] float32 в порядке POINT_LANDMARKS (доли кадра, NaN — часть не
найдена, как в parquet GISLR), absent [T, 4] — нет части в кадре (поза, левая кисть, правая кисть, лицо; нужно только
для «растекания» NaN по соседним индексам 543 в бикубическом ресэмплинге TF, см. resample).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# [c7] индексы 543 точек GISLR (лицо 0–467, левая кисть 468–488, поза 489–521, правая кисть 522–542)
NOSE = [1, 2, 98, 327]
LNOSE, RNOSE = [98], [327]
LIP = [0, 61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
       78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LLIP = [84, 181, 91, 146, 61, 185, 40, 39, 37, 87, 178, 88, 95, 78, 191, 80, 81, 82]
RLIP = [314, 405, 321, 375, 291, 409, 270, 269, 267, 317, 402, 318, 324, 308, 415, 310, 311, 312]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE
NUM_NODES = len(POINT_LANDMARKS)            # 118
CHANNELS = 6 * NUM_NODES                    # 708
MAX_LEN = 384
REF_POINT = 17                              # [c7] Preprocess: центр — точка лица 17 (губы)
assert NUM_NODES == 118 and len(set(POINT_LANDMARKS)) == 118

# [c19] CFG (8 ядер TPU: batch 64 × 8, lr 5e-4 × 8)
CFG = dict(max_len=384, replicas=8, lr=5e-4 * 8, weight_decay=0.1, lr_min=1e-6, epoch=300, warmup=0, batch_size=64 * 8,
           awp=True, awp_lambda=0.2, awp_start_epoch=15, dropout_start_epoch=15, decay_type="cosine", dim=192,
           label_smoothing=0.1, shuffle=32768, seeds=(42, 43, 44, 45, 46))  # 46 — пятая затравка турнира (0059 п. 5); ансамбль «как у автора» — 42–45


def part_of(k: int) -> int:
    """Индекс 543 → столбец absent: 0 поза, 1 левая кисть, 2 правая кисть, 3 лицо (порядок present Э0)."""
    return 3 if k < 468 else 1 if k < 489 else 0 if k < 522 else 2


LOCAL = {k: i for i, k in enumerate(POINT_LANDMARKS)}
OWN_PART = np.array([part_of(k) for k in POINT_LANDMARKS])
# Бикубика TF по оси «точки» (масштаб 1): веса отводов k−1, k, k+1, k+2 равны (0, 1, 0, 0), но 0 · NaN = NaN —
# точка получает NaN, если в отводах по времени нет части соседа. Для каждой точки — части соседей (кроме своей).
NEIGH_PARTS = [sorted({part_of(min(max(j, 0), 542)) for j in (k - 1, k + 1, k + 2)} - {part_of(k)}) for k in POINT_LANDMARKS]


def _flip_perm() -> np.ndarray:
    """[c8] flip_lr: перестановка 118 точек (кисти, губы, глаза, нос — пары L/R; поза в наших точках не участвует)."""
    perm = np.arange(NUM_NODES)
    for a, b in ((LHAND, RHAND), (LLIP, RLIP), (LEYE, REYE), (LNOSE, RNOSE)):
        for x, y in zip(a, b):
            perm[LOCAL[x]], perm[LOCAL[y]] = LOCAL[y], LOCAL[x]
    assert (perm[perm] == np.arange(NUM_NODES)).all()
    return perm


FLIP = _flip_perm()

# ---------------------------------------------------------------- UzSohn (reports/UZSOHN_prereg.md): флаги вариантов
# По умолчанию (pose=hand_gcn=parts=False) граф, порядок инициализации и вход — бит в бит прежний Sohn (tests/test_sohn.py).
POSE_MP4 = [11, 12, 13, 14]                 # идея 3: плечи и локти MediaPipe (в 543 GISLR — 489 + j); добавляются ПОСЛЕ 118
NUM_NODES_POSE = NUM_NODES + len(POSE_MP4)  # 122
EXTRA = 8 * 3 + 12 + 84                     # поза x, y, dx, dx2 (24) + запястья от носа/губы 17/середины плеч (12) + кисти в своей системе (84)
CHANNELS_POSE = CHANNELS + EXTRA            # 828
EXTRA_CLIP = 10.0
FLIP_POSE = np.concatenate([FLIP, NUM_NODES + np.array([1, 0, 3, 2])])   # flip_lr: 11 ↔ 12, 13 ↔ 14
HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12), (9, 13),
              (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]   # кисть MediaPipe
LH_LOC = [LOCAL[k] for k in LHAND]
RH_LOC = [LOCAL[k] for k in RHAND]


# ---------------------------------------------------------------- ресэмплинг по времени = tf.image.resize (TF 2.12)
def _tab_keys(d: np.ndarray, a: float = -0.5) -> np.ndarray:
    """Ядро Кейса (A = −0,5 при half_pixel_centers, resize_bicubic_op.cc), по таблице 1024 отсчётов."""
    d = np.abs(d)
    return np.where(d <= 1, ((a + 2) * d - (a + 3)) * d * d + 1, np.where(d < 2, ((a * d - 5 * a) * d + 8 * a) * d - 4 * a, 0.0))


def resize_time(x: np.ndarray, n: int, method: str, absent: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """x [T, P, C] → [n, P, C] как tf.image.resize(x, (n, P), method) (half_pixel_centers, без сглаживания).
    absent [T, 4] — для «растекания» NaN от соседей в бикубике; возвращается пересчитанным (nearest-выбор кадров /
    «любой отвод без части»)."""
    T = len(x)
    s = np.float32(T / n)
    o = np.arange(n, dtype=np.float32)
    if method == "nearest":           # HalfPixelScalerForNN: floor((o + 0.5) · s)
        i = np.minimum(np.floor((o + 0.5) * s).astype(np.int64), T - 1)
        return x[i], (absent[i] if absent is not None else None)
    src = (o + 0.5) * s - 0.5
    f = np.floor(src)
    if method == "bilinear":          # resize_bilinear_op.cc: top + (bottom − top) · lerp
        lo = np.maximum(f, 0).astype(np.int64)
        hi = np.minimum(np.ceil(src), T - 1).astype(np.int64)
        lerp = (src - f).astype(np.float32)[:, None, None]
        y = x[lo] + (x[hi] - x[lo]) * lerp
        return y, (absent[lo] | absent[hi] if absent is not None else None)
    assert method == "bicubic", method
    off = np.rint((src - f) * 1024).astype(np.int64)          # lrintf(delta · kTableSize)
    dq = off / 1024.0
    fi = f.astype(np.int64)
    taps = np.stack([fi - 1, fi, fi + 1, fi + 2], 1)          # [n, 4]
    w = np.stack([_tab_keys(1 + dq), _tab_keys(dq), _tab_keys(1 - dq), _tab_keys(2 - dq)], 1)
    ok = (taps >= 0) & (taps < T)
    w = np.where(ok, w, 0.0)                                   # отвод вне диапазона — вес 0 (Keys), затем нормировка
    ws = w.sum(1, keepdims=True)
    w = np.where(np.abs(ws) >= 1000 * np.finfo(np.float32).tiny, w / ws, w).astype(np.float32)
    tc = np.clip(taps, 0, T - 1)
    y = np.einsum("na,napc->npc", w, x[tc])                     # 0 · NaN = NaN, как в TF
    if absent is None:
        return y, None
    ab = absent[tc].any(1)                                      # [n, 4]: в каком-то отводе части нет
    for p, parts in enumerate(NEIGH_PARTS):
        if parts:
            y[ab[:, parts].any(1), p] = np.nan
    return y, ab


# ---------------------------------------------------------------- аугментации [c8]
def augment(x: np.ndarray, absent: np.ndarray, rng: np.random.Generator, max_len: int = MAX_LEN, flip: np.ndarray | None = None,
            extra: np.ndarray | None = None, extra_flip: list[int] | None = None):
    """augment_fn: resample (0,8) → flip_lr (0,5) → temporal_crop → affine (0,75) → temporal_mask (0,5) → spatial_mask (0,5).
    x [T, 118, 2] (NaN — нет точки). Порядок и вероятности — как у них; случайные числа — свой генератор.
    U33 (uzsign_ai_v2/reports/U33_prereg.md, A1 с рецептом Sohn): flip — своя перестановка точек (85 точек A1); extra [T, K] —
    покадровые величины (видимость позы A1): тот же ресэмплинг, перестановка extra_flip, обрезка и временная маска → (x, extra).
    По умолчанию (flip=extra=None) — прежняя функция бит в бит (тот же расход случайных чисел)."""
    u = rng.random
    if u() < 0.8:                                    # resample(x, (0.5, 1.5)), interp1d_ 'random'
        rate = np.float32(rng.uniform(0.5, 1.5))
        n = max(1, int(np.float32(rate * np.float32(len(x)))))
        m = "bilinear" if u() < 0.33 else ("bicubic" if u() < 0.5 else "nearest")
        x, absent = resize_time(x, n, m, absent)
        if extra is not None:
            extra = resize_time(extra[:, :, None], n, m)[0][..., 0]
    x = x.copy()
    if extra is not None:
        extra = extra.copy()
    if u() < 0.5:                                    # flip_lr: x = 1 − x, обмен L/R
        x[..., 0] = 1 - x[..., 0]
        x = x[:, (FLIP if x.shape[1] == NUM_NODES else FLIP_POSE) if flip is None else flip]   # UzSohn идея 3: + поза (точки 118–121)
        if extra is not None:
            extra = extra[:, extra_flip]
    L = len(x)                                       # temporal_crop(x, max_len)
    off = int(rng.integers(0, int(np.clip(L - max_len, 1, max_len))))
    x = x[off:off + max_len]
    if extra is not None:
        extra = extra[off:off + max_len]
    if u() < 0.75:                                   # spatial_random_affine
        center = np.array([0.5, 0.5], np.float32)
        x = x * np.float32(rng.uniform(0.8, 1.2))
        sx = sy = np.float32(rng.uniform(-0.15, 0.15))
        if u() < 0.5:
            sx = np.float32(0.0)
        else:
            sy = np.float32(0.0)
        x = x @ np.array([[1.0, sx], [sy, 1.0]], np.float32)
        center = center + np.array([sy, sx], np.float32)
        r = np.float32(rng.uniform(-30, 30)) / 180 * np.float32(np.pi)
        c, s = np.cos(r), np.sin(r)
        x = (x - center) @ np.array([[c, s], [-s, c]], np.float32) + center
        x = x + np.float32(rng.uniform(-0.1, 0.1))   # один сдвиг на обе оси (как у них)
    if u() < 0.5:                                    # temporal_mask(size=(0.2, 0.4))
        L = len(x)
        ms = int(np.float32(L) * np.float32(rng.uniform(0.2, 0.4)))
        mo = int(rng.integers(0, int(np.clip(L - ms, 1, L))))
        x[mo:mo + ms] = np.nan
        if extra is not None:
            extra[mo:mo + ms] = np.nan
    if u() < 0.5:                                    # spatial_mask(size=(0.2, 0.4))
        oy, ox, sz = np.float32(u()), np.float32(u()), np.float32(rng.uniform(0.2, 0.4))
        with np.errstate(invalid="ignore"):
            m = (ox < x[..., 0]) & (x[..., 0] < ox + sz) & (oy < x[..., 1]) & (x[..., 1] < oy + sz)
        x[m] = np.nan
    return x.astype(np.float32) if extra is None else (x.astype(np.float32), extra.astype(np.float32))


def filter_nans(x: np.ndarray, absent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[c8] filter_nans_tf: убрать кадры, где все 118 точек NaN. Пустая запись → один кадр NaN (у них длина 0 даёт
    NaN в потере; у нас таких записей счётчик в манифесте, расхождение № 9 предрегистрации)."""
    keep = ~np.isnan(x).all((1, 2))
    if not keep.any():
        return np.full((1, x.shape[1], 2), np.nan, np.float32), np.ones((1, 4), bool)
    return x[keep], absent[keep]


def center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Центр (nan-среднее точки 17, нет — 0,5) и масштаб (nan-СКО 118 точек вокруг центра по осям) Preprocess [c7]."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ref = x[:, LOCAL[REF_POINT]]
        cnt = (~np.isnan(ref)).sum(0)
        mean = np.where(cnt > 0, np.nansum(ref, 0) / np.maximum(cnt, 1), np.nan)
        mean = np.where(np.isnan(mean), np.float32(0.5), mean).astype(np.float32)
        d = x - mean
        n2 = (~np.isnan(d)).sum((0, 1))
        std = np.sqrt(np.nansum(d * d, (0, 1)) / n2).astype(np.float32)   # 0/0 → NaN, как tf_nan_std
    return mean, std


def preprocess_pose(x: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """UzSohn идея 3: x [T, 122, 2] (118 Sohn + плечи/локти) → [T, 828] = preprocess(118 точек) (бит в бит) + 120 признаков
    в той же нормировке Sohn (центр и масштаб — по 118 точкам): поза x, y, dx, dx2 (4 точки); запястья L/R минус нос (1),
    нижняя губа (17) и середина плеч в кадре, / ширина плеч (nan-среднее по записи); 21 точка каждой кисти в своей системе
    (точка − запястье) / |точка 9 − запястье| (Н2-кисть, без поворота). NaN → 0, признаки 120 — в [−10; 10]."""
    base = preprocess(x[:, :NUM_NODES], max_len)
    mean, std = center_scale(x[:, :NUM_NODES])
    with np.errstate(invalid="ignore", divide="ignore"):
        z = ((x - mean) / std)[:max_len]
        T = len(z)
        pose = z[:, NUM_NODES:]
        dp, dp2 = np.zeros_like(pose), np.zeros_like(pose)
        if T > 1:
            dp[:-1] = pose[1:] - pose[:-1]
        if T > 2:
            dp2[:-2] = pose[2:] - pose[:-2]
        sl, sr = pose[:, 0], pose[:, 1]
        sw = np.linalg.norm(sl - sr, axis=-1)
        w = np.nanmean(sw) if np.isfinite(sw).any() else np.nan
        mid = (sl + sr) / 2
        rel = [(z[:, h] - r) / w for h in (LH_LOC[0], RH_LOC[0]) for r in (z[:, LOCAL[1]], z[:, LOCAL[17]], mid)]
        loc = []
        for H in (LH_LOC, RH_LOC):
            q = z[:, H]
            s = np.linalg.norm(q[:, 9] - q[:, 0], axis=-1)
            loc.append((q - q[:, :1]) / s[:, None, None])
        ext = np.concatenate([pose.reshape(T, -1), dp.reshape(T, -1), dp2.reshape(T, -1), np.stack(rel, 1).reshape(T, -1),
                              loc[0].reshape(T, -1), loc[1].reshape(T, -1)], 1)
    ext = np.clip(np.nan_to_num(ext, nan=0.0, posinf=0.0, neginf=0.0), -EXTRA_CLIP, EXTRA_CLIP)
    assert ext.shape[1] == EXTRA
    return np.concatenate([base, ext.astype(np.float32)], 1)


# ---------------------------------------------------------------- UzSohn этап 1 (reports/UZSOHN_prereg.md): вход U02 / U01a / U04
# Токены входа: u02s — масштаб = ширина плеч (один на обе оси, в пикселях: y · H/W); u02c — центр = середина плеч;
# U02 — два плеча: U02-1 = u02s, затем U02-2 = + u02c (к1 0124 п. 2, к3 0032 п. 2); u01a — +4 точки позы (11–14) во вход
# Sohn (x, y, dx, dx2: 732 столбца), отражение 11↔12, 13↔14; u04 — на кисть 210 попарных расстояний / медиана ладони |9 − 0|
# по кадрам + 15 значений 1 + cos углов суставов (без dx), +450 столбцов.
# Запись x [T, 122, 2] = 118 точек Sohn + поза 11–14 (поза нужна и без u01a — для u02*). filter_nans — по 118 (как у Sohn).
INPUT_TOKENS = ("u02s", "u02c", "u01a", "u04")
FINGERS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
PAIRS = np.array([(i, j) for i in range(21) for j in range(i + 1, 21)])   # 210
HAND_FEATS = 2 * (len(PAIRS) + 15)                                        # 450


def input_channels(tokens) -> int:
    return 6 * (NUM_NODES_POSE if "u01a" in tokens else NUM_NODES) + (HAND_FEATS if "u04" in tokens else 0)


def hand_feats(xp: np.ndarray) -> np.ndarray:
    """xp [T, 118, 2] в изотропных координатах (пиксели) → [T, 450]: на кисть Л, П — 210 расстояний / ладонь + 15 значений
    1 + cos углов (README 2-го места GISLR: угол <a,b,c> в вершине b цепочки пальца между лучами к a и к c; к3 0030 п. 5,
    к1 0122 п. 3; 1 + cos — к1 0123 п. 19, к3 0031: у bf16 stem cos ≈ −1 сливается при сгибе < 3,6°; прямой палец — 0,
    прямой угол — 1, обрезка [0; 2] снимает выход за край на шаг float32: < 0 — на ≥ 1,2·10⁻⁷, > 2 — на ≥ 2,4·10⁻⁷).
    Ладонь — медиана |9 − 0| по кадрам xp, где есть точки 0 и 9 этой кисти (к1 0123 п. 15, к3 0031 п. 15: по кадрам, что видит
    вход, — при обучении после масок); таких кадров нет — 210 расстояний этой кисти 0, как без кисти (1 + cos — по кадру). Нет
    точки — 0; обрезка [−10; 10]."""
    out = []
    with np.errstate(invalid="ignore", divide="ignore"):
        for H in (LH_LOC, RH_LOC):
            q = xp[:, H]
            palm = np.linalg.norm(q[:, 9] - q[:, 0], axis=-1)
            med = np.nanmedian(palm) if np.isfinite(palm).any() else np.nan   # nanmedian пустого — предупреждение
            d = np.linalg.norm(q[:, PAIRS[:, 0]] - q[:, PAIRS[:, 1]], axis=-1) / med
            ang = []
            for f in FINGERS:
                for a, b, c in zip(f, f[1:], f[2:]):
                    u, v = q[:, a] - q[:, b], q[:, c] - q[:, b]
                    ang.append(np.clip(1 + (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), 0, 2))
            out += [d, np.stack(ang, 1)]
    f = np.concatenate(out, 1)
    return np.clip(np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0), -EXTRA_CLIP, EXTRA_CLIP).astype(np.float32)


def center_scale_u(x: np.ndarray, aspect: float, tokens, stat: str = "median") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (k, mean, std): z = (x · k − mean) / std. По умолчанию (нет u02*) — center_scale Sohn по 118 точкам, k = (1, 1).
    u02s (U02-1): k = (1, H/W), std = медиана по кадрам ширины плеч на обе оси; центр — u02c: медиана середины плеч, иначе
    центр Sohn (среднее точки 17, нет — 0,5) в тех же координатах. Только u02c (U02-2 без U02-1): центр — середина плеч,
    масштаб — СКО Sohn (по осям, вокруг точки 17): чистый сдвиг опоры (к3 0018) для видимых точек; пропуски (NaN → 0 после
    центрирования) у u02c означают середину плеч, у Sohn и u02s — губу 17. Опора — по самому x, при обучении — после
    temporal/spatial_mask, как у Sohn (вариант (в): к1 0124 п. 3, к3 0032 п. 3): нет кадра с обоими плечами — центр и масштаб
    Sohn, k = (1, 1), вход без u01a и u04 = Sohn бит в бит. stat = mean — только печать КВ ширины лица (uzsohn_gates.py
    u02stats); вход — всегда медиана (к1 0123 п. 2, к3 0031 п. 2)."""
    k = np.array([1.0, aspect if "u02s" in tokens else 1.0], np.float32)
    agg = np.nanmedian if stat == "median" else np.nanmean
    p = x[:, NUM_NODES:NUM_NODES + 2] * k   # плечи 11, 12
    ok = np.isfinite(p).all((1, 2))
    mean, std = center_scale(x[:, :NUM_NODES])
    if not ({"u02s", "u02c"} & set(tokens)) or not ok.any():
        return np.ones(2, np.float32), mean, std
    mean = (mean * k).astype(np.float32)                              # центр Sohn в координатах k
    if "u02c" in tokens:
        mean = np.nanmedian((p[ok, 0] + p[ok, 1]) / 2, 0).astype(np.float32)
    if "u02s" in tokens:
        w = np.float32(agg(np.linalg.norm(p[ok, 0] - p[ok, 1], axis=-1)))
        std = np.array([w, w], np.float32)
    return k, mean, std


def preprocess_u(x: np.ndarray, aspect: float, tokens, max_len: int = MAX_LEN) -> np.ndarray:
    """UzSohn этап 1: x [T, 122, 2] → [T, input_channels(tokens)]. Без токенов = preprocess(x[:, :118]) бит в бит."""
    if not tokens:
        return preprocess(x[:, :NUM_NODES], max_len)
    k, mean, std = center_scale_u(x, aspect, tokens)
    P = NUM_NODES_POSE if "u01a" in tokens else NUM_NODES
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (x[:, :P] * k - mean) / std
    z = z[:max_len]
    T = len(z)
    dx, dx2 = np.zeros_like(z), np.zeros_like(z)
    if T > 1:
        dx[:-1] = z[1:] - z[:-1]
    if T > 2:
        dx2[:-2] = z[2:] - z[:-2]
    out = [np.nan_to_num(np.concatenate([z.reshape(T, -1), dx.reshape(T, -1), dx2.reshape(T, -1)], 1), nan=0.0)]
    if "u04" in tokens:
        out.append(hand_feats(x[:max_len, :NUM_NODES] * np.array([1.0, aspect], np.float32)))
    return np.concatenate(out, 1).astype(np.float32)


def preprocess(x: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """[c7] Preprocess: центр — nan-среднее точки 17 по кадрам (нет — 0,5), масштаб — nan-СКО всех 118 точек вокруг
    центра (по осям отдельно), окно — первые max_len кадров, x, y + dx (шаг 1) + dx2 (шаг 2), NaN → 0. → [T, 708]."""
    mean, std = center_scale(x)
    with np.errstate(invalid="ignore", divide="ignore"):
        x = (x - mean) / std
    x = x[:max_len]
    T = len(x)
    dx = np.zeros_like(x)
    dx2 = np.zeros_like(x)
    if T > 1:
        dx[:-1] = x[1:] - x[:-1]
    if T > 2:
        dx2[:-2] = x[2:] - x[:-2]
    out = np.concatenate([x.reshape(T, -1), dx.reshape(T, -1), dx2.reshape(T, -1)], 1)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


# ---------------------------------------------------------------- модель [c14–c16]
def glorot_(w: torch.Tensor, fan_in: int, fan_out: int) -> None:
    lim = math.sqrt(6.0 / (fan_in + fan_out))
    nn.init.uniform_(w, -lim, lim)


def dense(i: int, o: int, bias: bool) -> nn.Linear:
    """keras Dense: ядро glorot_uniform, смещение 0."""
    m = nn.Linear(i, o, bias=bias)
    glorot_(m.weight, i, o)
    if bias:
        nn.init.zeros_(m.bias)
    return m


class MaskedBN(nn.Module):
    """keras BatchNormalization(momentum=0.95, epsilon=1e-3) по последней оси с маской шагов (Keras ≥ 2.12: моменты
    только по шагам маски, дисперсия смещённая), в float32. Бегущие средние обновляются commit(): за шаг по каждому
    «проходу» — среднее по 8 частям (их 8 ядер TPU: переменные ON_READ, агрегация MEAN; у AWP два прохода за шаг)."""

    def __init__(self, c: int, momentum: float = 0.95, eps: float = 1e-3):
        super().__init__()
        self.weight, self.bias = nn.Parameter(torch.ones(c)), nn.Parameter(torch.zeros(c))
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))
        self.momentum, self.eps = momentum, eps
        self.acc: dict[int, list] = {}
        self.pass_id = 0

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if self.training:
            w = m[..., None].float()
            n = w.sum()
            mean = (x * w).sum((0, 1)) / n
            var = (((x - mean) ** 2) * w).sum((0, 1)) / n
            self.acc.setdefault(self.pass_id, []).append((mean.detach(), var.detach()))
        else:
            mean, var = self.running_mean, self.running_var
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias

    @torch.no_grad()
    def commit(self) -> None:
        for k in sorted(self.acc):
            mean = torch.stack([a for a, _ in self.acc[k]]).mean(0)
            var = torch.stack([b for _, b in self.acc[k]]).mean(0)
            self.running_mean.mul_(self.momentum).add_(mean * (1 - self.momentum))
            self.running_var.mul_(self.momentum).add_(var * (1 - self.momentum))
        self.acc = {}


def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """keras GlobalAveragePooling1D с маской: Σ(x·m) / Σm по времени."""
    w = m[..., None].to(x.dtype)
    return (x * w).sum(1) / w.sum(1)


def sample_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """keras Dropout(p, noise_shape=(None, 1, 1)): вся ветвь записи обнуляется, остальное × 1/(1 − p)."""
    if not training or p == 0:
        return x
    keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= p).to(x.dtype)
    return x * keep / (1 - p)


class ECA(nn.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)
        glorot_(self.conv.weight, k, k)                  # Conv1D(1, 5): ядро [5, 1, 1] → fan_in = fan_out = 5

    def forward(self, x, m):
        s = self.conv(masked_mean(x, m)[:, None, :])[:, 0]
        return x * torch.sigmoid(s)[:, None, :]


class Conv1DBlock(nn.Module):
    """[c14] расширение ×2 (swish) → причинная depthwise-свёртка 17 → BN → ECA → проекция → Dropout(записи) → + вход."""

    def __init__(self, dim: int, k: int = 17, drop: float = 0.2, expand: int = 2):
        super().__init__()
        e = dim * expand
        self.expand = dense(dim, e, True)
        self.dw = nn.Conv1d(e, e, k, groups=e, bias=False)
        glorot_(self.dw.weight, k * e, k)               # DepthwiseConv1D: ядро [17, C, 1] → fan_in 17·C, fan_out 17
        self.k = k
        self.bn = MaskedBN(e)
        self.eca = ECA()
        self.project = dense(e, dim, True)
        self.drop = drop

    def forward(self, x, m):
        h = F.silu(self.expand(x))
        ht = h.transpose(1, 2)
        h = self.dw(F.pad(ht, (self.k - 1, 0)))
        for c in getattr(self, "dw_ms", ()):            # UzSohn идея 5: + причинные depthwise 5 и 11 (нули в начале)
            h = h + c(F.pad(ht, (c.kernel_size[0] - 1, 0)))
        h = h.transpose(1, 2)
        h = self.eca(self.bn(h, m), m)
        return sample_dropout(self.project(h), self.drop, self.training) + x


class MHSA(nn.Module):
    """[c15] qkv без смещения → [B, T, H, 3·dim/H] → split q, k, v; масштаб dim^−0,5 (не головы); маска ключей −1e9."""

    def __init__(self, dim: int = 192, heads: int = 4, drop: float = 0.2):
        super().__init__()
        self.dim, self.h, self.scale = dim, heads, dim ** -0.5
        self.qkv = dense(dim, 3 * dim, False)
        self.proj = dense(dim, dim, False)
        self.drop = nn.Dropout(drop)

    def forward(self, x, m):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, self.h, -1).permute(0, 2, 1, 3).split(self.dim // self.h, -1)
        a = (q @ k.transpose(-1, -2)) * self.scale
        a = a + (~m)[:, None, None, :].to(a.dtype) * -1e9
        a = self.drop(torch.softmax(a.float(), -1).to(v.dtype))
        return self.proj((a @ v).permute(0, 2, 1, 3).reshape(B, T, self.dim))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 192, expand: int = 2, drop: float = 0.2):
        super().__init__()
        self.bn1, self.attn = MaskedBN(dim), MHSA(dim)
        self.bn2 = MaskedBN(dim)
        self.fc1, self.fc2 = dense(dim, dim * expand, False), dense(dim * expand, dim, False)
        self.drop = drop

    def forward(self, x, m):
        x = x + sample_dropout(self.attn(self.bn1(x, m), m), self.drop, self.training)
        h = self.fc2(F.silu(self.fc1(self.bn2(x, m))))
        return x + sample_dropout(h, self.drop, self.training)


def _cols(points: list[int]) -> list[int]:
    """Столбцы входа 708 точек (x, y, dx, dx2 — блоки по 236) для локальных индексов points."""
    return [k * 2 * NUM_NODES + 2 * p + a for k in range(3) for p in points for a in (0, 1)]


def _zero(m: nn.Linear) -> nn.Linear:
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m


class HandGCN(nn.Module):
    """UzSohn идея 2: покадровый граф кисти до свёртки по времени. Узел — 6 признаков Sohn точки кисти (x, y, dx, dx2);
    3 слоя H ← H + SiLU(Â H W) (Â = D^−½(A + I)D^−½ по 21 суставу, общие для двух кистей; 1-й слой без остатка), ширина 64;
    выход — Dense(21·64 → dim) своя на каждую кисть, нулевая инициализация (в начале вклад 0); прибавляется к stem."""

    def __init__(self, dim: int, width: int = 64, layers: int = 3):
        super().__init__()
        A = torch.eye(21)
        for a, b in HAND_EDGES:
            A[a, b] = A[b, a] = 1
        d = A.sum(1).rsqrt()
        self.register_buffer("A", d[:, None] * A * d[None], persistent=False)
        self.register_buffer("idx", torch.tensor([_cols([p]) for p in LH_LOC] + [_cols([p]) for p in RH_LOC]), persistent=False)
        self.w = nn.ModuleList([dense(6, width, False)] + [dense(width, width, False) for _ in range(layers - 1)])
        self.out = nn.ModuleList([_zero(nn.Linear(21 * width, dim, bias=False)) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        g = x[..., self.idx].view(B, T, 2, 21, 6)       # [_cols] на точку: (x, y), (dx…), (dx2…) — 6 признаков
        res = 0
        for k, out in enumerate(self.out):
            h = g[:, :, k]
            for i, w in enumerate(self.w):
                y = F.silu(torch.einsum("uv,btvc->btuc", self.A.to(h.dtype), w(h)))
                h = y if i == 0 else h + y
            res = res + out(h.reshape(B, T, -1))
        return res


class HandCross(nn.Module):
    """UzSohn идея 4: перекрёстное внимание потоков кистей по времени. Поток кисти — Dense(126 → 64) её 21 точки (x, y, dx,
    dx2); Л спрашивает П и П спрашивает Л (4 головы, ключи по маске); выход Dense(128 → dim), нули в начале; прибавляется к stem."""

    def __init__(self, dim: int, d: int = 64, heads: int = 4):
        super().__init__()
        self.register_buffer("idx", torch.tensor([_cols(LH_LOC), _cols(RH_LOC)]), persistent=False)
        self.inp = nn.ModuleList([dense(126, d, True) for _ in range(2)])
        self.q, self.kv = dense(d, d, False), dense(d, 2 * d, False)
        self.h, self.d = heads, d
        self.out = _zero(nn.Linear(2 * d, dim, bias=False))

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        s = [F.silu(f(x[..., self.idx[k]])) for k, f in enumerate(self.inp)]

        def att(a, b):
            q = self.q(a).view(B, T, self.h, -1).transpose(1, 2)
            k, v = self.kv(b).view(B, T, self.h, -1).transpose(1, 2).chunk(2, -1)
            w = (q @ k.transpose(-1, -2)) * (self.d // self.h) ** -0.5 + (~m)[:, None, None, :].to(q.dtype) * -1e9
            return (torch.softmax(w.float(), -1).to(v.dtype) @ v).transpose(1, 2).reshape(B, T, self.d)

        return self.out(torch.cat([att(s[0], s[1]), att(s[1], s[0])], -1))


class SohnNet(nn.Module):
    """[c16] get_model(dim=192): stem Dense + BN → (3 × Conv1DBlock + TransformerBlock) × 2 → Dense(2·dim) → среднее по
    маске → LateDropout(0,8) → классификатор. late_dropout_on включает LateDropout (с шага dropout_start_epoch·spe).
    UzSohn (reports/UZSOHN_prereg.md): pose — вход 828 (stem_extra 120 → dim, нули в начале); hand_gcn — HandGCN к stem;
    parts — stem по частям (губы, кисть Л, кисть П, нос + глаза, [поза]): часть = срез весов stem + своя остаточная MLP
    (2 × Dense dim, выход — нули), слияние — сумма. Новые модули создаются ПОСЛЕ прежних (та же инициализация прежних при
    той же затравке) и в начале не меняют выход (вариант с весами Sohn = Sohn; tests/test_sohn.py). Идеи 4–7: hand_cross
    (HandCross к stem), ms_conv (+ depthwise 5 и 11 в каждом Conv1DBlock), attn_pool (внимание вместо среднего), cos_head
    (косинусная голова — меняет выход с начала); aux_shape (идея 8) — заготовка, NotImplementedError."""

    def __init__(self, n_classes: int, dim: int = 192, channels: int = CHANNELS, pose: bool = False, hand_gcn: bool = False,
                 parts: bool = False, hand_cross: bool = False, ms_conv: bool = False, attn_pool: bool = False,
                 cos_head: bool = False, aux_shape: bool = False):
        super().__init__()
        self.stem = dense(channels, dim, False)
        self.stem_bn = MaskedBN(dim)
        layers = []
        for _ in range(4 if dim == 384 else 2):
            layers += [Conv1DBlock(dim), Conv1DBlock(dim), Conv1DBlock(dim), TransformerBlock(dim)]
        self.blocks = nn.ModuleList(layers)
        self.top = dense(dim, 2 * dim, True)
        self.classifier = dense(2 * dim, n_classes, True)
        self.late_dropout_on = False
        self.pose, self.hand_gcn, self.parts = pose, hand_gcn, parts
        self.hand_cross, self.attn_pool, self.cos_head = hand_cross, attn_pool, cos_head
        if aux_shape:   # идея 8 — заготовка: нужна разметка формы кисти / буквы для классов УзЖЯ (решение автора)
            raise NotImplementedError("UzSohn идея 8 (вспомогательная голова формы кисти): разметки нет — заготовка")
        if pose:
            self.stem_extra = _zero(nn.Linear(EXTRA, dim, bias=False))
        if hand_gcn:
            self.gcn = HandGCN(dim)
        if parts:
            ext = (lambda a, b: list(range(a, b)) if pose else [])
            groups = [(LIP, []), (LHAND, ext(36, 78)), (RHAND, ext(78, 120)), (NOSE + REYE + LEYE, [])]
            if pose:
                groups.append(([], list(range(36))))  # поза + запястья относительно тела
            for i, (g, e) in enumerate(groups):
                self.register_buffer(f"part_base{i}", torch.tensor(_cols([LOCAL[k] for k in g]), dtype=torch.long), persistent=False)
                self.register_buffer(f"part_ext{i}", torch.tensor(e, dtype=torch.long), persistent=False)
            self.n_parts = len(groups)
            assert sorted(torch.cat([getattr(self, f"part_base{i}") for i in range(self.n_parts)]).tolist()) == list(range(CHANNELS))
            assert sorted(sum((e for _, e in groups), [])) == (list(range(EXTRA)) if pose else [])
            self.part_mlp = nn.ModuleList([nn.Sequential(dense(dim, dim, True), nn.SiLU(), _zero(nn.Linear(dim, dim))) for _ in groups])
        if hand_cross:
            self.cross = HandCross(dim)
        if ms_conv:     # идея 5: модули создаются здесь (после прежних), а подключаются к каждому Conv1DBlock
            for b in self.blocks:
                if isinstance(b, Conv1DBlock):
                    e = b.dw.in_channels
                    b.dw_ms = nn.ModuleList([nn.Conv1d(e, e, k, groups=e, bias=False) for k in (5, 11)])
                    for c in b.dw_ms:
                        nn.init.zeros_(c.weight)
        if attn_pool:   # идея 6: внимание по времени; нули в начале = среднее по маске (как GAP)
            self.pool_score = _zero(nn.Linear(2 * dim, 1))
        if cos_head:    # идея 7: косинусная голова — строки classifier.weight = прототипы классов, масштаб обучаемый
            self.cos_scale = nn.Parameter(torch.tensor(16.0))

    def stem_in(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.pose or self.hand_gcn or self.parts or self.hand_cross):
            return self.stem(x)
        xb = x[..., :CHANNELS]
        if self.parts:
            h = 0
            for i, mlp in enumerate(self.part_mlp):
                b, e = getattr(self, f"part_base{i}"), getattr(self, f"part_ext{i}")
                hp = F.linear(xb[..., b], self.stem.weight[:, b]) if len(b) else 0
                if self.pose and len(e):
                    hp = hp + F.linear(x[..., CHANNELS + e], self.stem_extra.weight[:, e])
                h = h + hp + mlp(hp)
        else:
            h = self.stem(xb)
            if self.pose:
                h = h + self.stem_extra(x[..., CHANNELS:])
        if self.hand_gcn:
            h = h + self.gcn(xb)
        return h


    def features(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        x = x * m[..., None].to(x.dtype)                 # keras Masking: шаги вне маски = 0
        h = self.stem_in(x)
        if self.hand_cross:
            h = h + self.cross(x[..., :CHANNELS], m)
        h = self.stem_bn(h, m)   # BN — float32; при autocast bf16 остальное приводит autocast
        for b in self.blocks:
            h = b(h, m)
        h = self.top(h)
        if self.attn_pool:
            a = self.pool_score(h).float()[..., 0] + (~m).float() * -1e9
            return (torch.softmax(a, 1)[..., None].to(h.dtype) * h).sum(1)
        return masked_mean(h, m)

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        f = self.features(x, m)
        if self.training and self.late_dropout_on:
            f = F.dropout(f, 0.8, True)
        if self.cos_head:
            return self.cos_scale * F.normalize(f.float(), dim=-1) @ F.normalize(self.classifier.weight.float(), dim=-1).T
        return self.classifier(f)

    def bns(self) -> list[MaskedBN]:
        return [b for b in self.modules() if isinstance(b, MaskedBN)]

    def set_pass(self, k: int) -> None:
        for b in self.bns():
            b.pass_id = k

    def commit_bn(self) -> None:
        for b in self.bns():
            b.commit()


def keras_param_count(model: SohnNet) -> dict[str, int]:
    """Счёт как в model.summary() Keras: обучаемые + бегущие средние BN + 1 счётчик LateDropout."""
    tr = sum(p.numel() for p in model.parameters())
    nt = sum(b.running_mean.numel() + b.running_var.numel() for b in model.bns()) + 1
    return {"trainable": tr, "non_trainable": nt, "total": tr + nt}


# ---------------------------------------------------------------- рецепт [c18, c19], tf_utils, tfa
def one_cycle_lr(step: int, lr: float, total_steps: int, lr_min: float) -> float:
    """tf_utils.schedules.OneCycleLR при warmup 0, sustain 0, decay_epochs = epochs, косинус."""
    s = min(float(step), float(total_steps))
    return 0.5 * (lr - lr_min) * (1 + math.cos(3.14159265359 * s / total_steps)) + lr_min


class RAdamLookahead:
    """tfa.optimizers.Lookahead(RectifiedAdam(lr=sched, weight_decay=sched·0,1, sma_threshold=4), sync_period=5):
    β 0,9 / 0,999, ε 1e-7; затухание весов в обновлении: var −= lr·(u + wd·var), wd = lr_sched·0,1 (на ВСЕ переменные);
    Lookahead: медленные веса = начальные, каждые 5 шагов slow += 0,5·(fast − slow), fast = slow."""

    def __init__(self, params, sma_threshold: float = 4.0, sync_period: int = 5, alpha: float = 0.5,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-7):
        self.params = [p for p in params if p.requires_grad]
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.slow = [p.detach().clone() for p in self.params]
        self.t, self.thr, self.k, self.alpha, self.b1, self.b2, self.eps = 0, sma_threshold, sync_period, alpha, b1, b2, eps

    @torch.no_grad()
    def step(self, lr: float, wd: float) -> None:
        self.t += 1
        f = np.float32                                   # коэффициенты — в float32, как _prepare_local tfa (важно у порога SMA)
        t, b1, b2 = f(self.t), f(self.b1), f(self.b2)
        b1p, b2p = b1 ** t, b2 ** t
        r1, r2 = f(1) / (f(1) - b1p), f(1) / (f(1) - b2p)
        sma_inf = f(2) / (f(1) - b2) - f(1)
        sma_t = sma_inf - f(2) * t * b2p * r2
        rect = bool(sma_t >= f(self.thr))
        r_t = float(np.sqrt((sma_t - f(4)) / (sma_inf - f(4)) * (sma_t - f(2)) / (sma_inf - f(2)) * sma_inf / sma_t)) if rect else 0.0
        r1, r2, lr, wd = float(r1), float(r2), float(f(lr)), float(f(wd))
        sync = self.t % self.k == 0
        for p, m, v, s in zip(self.params, self.m, self.v, self.slow):
            g = p.grad if p.grad is not None else torch.zeros_like(p)
            m.mul_(self.b1).add_(g, alpha=1 - self.b1)
            v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            mc = m * r1
            u = r_t * mc / ((v * r2).sqrt() + self.eps) if rect else mc
            p.sub_(lr * (u + wd * p))
            if sync:
                s.add_(p - s, alpha=self.alpha)
                p.copy_(s)

    def state_dict(self) -> dict:
        return {"t": self.t, "m": self.m, "v": self.v, "slow": self.slow}


def awp_perturb(params: list[torch.Tensor], grads: list[torch.Tensor], delta: float = 0.2) -> list[torch.Tensor]:
    """tf_utils.learners.AWP (eps = 0): каждому тензору + δ·g/‖g‖ (divide_no_nan: ‖g‖ = 0 → 0). Возвращает сдвиги.
    Без синхронизации с ГП (нормы — одним стеком)."""
    with torch.no_grad():
        n = torch.stack(torch._foreach_norm(grads))
        s = torch.where(n > 0, delta / n, torch.zeros_like(n))
        out = [g * s[i] for i, g in enumerate(grads)]
        torch._foreach_add_(params, out)
    return out


def collate(xs: list[np.ndarray], pad_to: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """Записи [T_i, 708] → x [B, T, 708], маска [B, T]. У них — padded_batch до max_len 384 значением −100 и Masking;
    здесь — до максимума части (кратно pad_to): шаги вне маски не влияют на шаги маски (причинная свёртка, маска во
    внимании, BN и средних), поэтому длина дополнения на выход не влияет (тест test_padding_invariance)."""
    T = max(len(a) for a in xs)
    T = int(math.ceil(T / pad_to) * pad_to)
    x = np.zeros((len(xs), T, xs[0].shape[1]), np.float32)
    m = np.zeros((len(xs), T), bool)
    for i, a in enumerate(xs):
        x[i, :len(a)], m[i, :len(a)] = a, True
    return torch.from_numpy(x), torch.from_numpy(m)


def schedules(step: int, total_steps: int, cfg: dict = CFG) -> tuple[float, float]:
    """[c18] lr = OneCycleLR(lr, …, lr_min); weight_decay = OneCycleLR(lr·wd, …, lr_min·wd) — оба от шага (iterations)."""
    lr = one_cycle_lr(step, cfg["lr"], total_steps, cfg["lr_min"])
    wd = one_cycle_lr(step, cfg["lr"] * cfg["weight_decay"], total_steps, cfg["lr_min"] * cfg["weight_decay"])
    return lr, wd


def train_step(model: SohnNet, opt: RAdamLookahead, parts: list, lr: float, wd: float, awp_on: bool,
               awp_lambda: float = 0.2, ls: float = 0.1, amp: bool = False, loss_fn=None) -> float:
    """Один шаг их model.fit на 8 ядрах TPU: parts — 8 частей пакета (x, маска, метки) = по одной на ядро.
    Потеря части / число частей (= среднее по пакету); BN — статистика по части; AWP (с шага awp_start) — на каждом
    ядре свой сдвиг весов по градиенту своей части (TPUMirroredVariable в контексте TPU), второй проход в сдвинутой
    точке, возврат весов; затем один шаг Lookahead(RAdam) по сумме градиентов. Возвращает среднюю потерю (1-й проход).
    U33: loss_fn(*часть) — своя потеря части (A1: часть (x, m, aspect, y)); проходы MaskedBN — только у модели с set_pass."""
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = None
    n, tot = len(parts), 0.0
    dev = params[0].device.type

    def loss_of(*p):
        with torch.autocast(dev, torch.bfloat16, enabled=amp):
            if loss_fn is not None:
                return loss_fn(*p) / n
            x, m, y = p
            return F.cross_entropy(model(x, m).float(), y, label_smoothing=ls) / n

    passes = hasattr(model, "set_pass")
    for p in parts:
        if passes:
            model.set_pass(0)
        loss = loss_of(*p)
        tot = tot + loss.detach() * n
        if awp_on:
            g = torch.autograd.grad(loss, params)
            d = awp_perturb(params, g, awp_lambda)
            if passes:
                model.set_pass(1)
            loss_of(*p).backward()
            with torch.no_grad():
                torch._foreach_sub_(params, d)
        else:
            loss.backward()
    opt.step(lr, wd)
    if passes:
        model.commit_bn()
    return float(tot) / n
