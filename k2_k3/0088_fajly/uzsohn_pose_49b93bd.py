"""UzSohn идея 3 (reports/UZSOHN_prereg.md): поза (плечи 11, 12, локти 13, 14) для входа Sohn — кадр в кадр с кэшами Sohn.

cd osnovnaya_model && PYTHONPATH=src:../uzsign_e1/src:../uzsign_ai_v2/src python scripts/uzsohn_pose.py [alluz] [alluz_x10] [slovo] [bukva] [uzsl]
slovo, bukva (UzSohn этап 1, предобучение S+B с позой): кэш МПК cache/<набор>.npz, кадры [lo, hi) записи → m13_build.to30 →
cache/UZSOHN/<набор>_pose.npz; проверка кадров — запястья кистей МПК = запястья кэша Sohn (float16, |Δ| < 2e-3).
uzsl (UzSohn в Э1 на корпусе, поправка 4 к TURNIR_prereg, 22.10 п. 1) — так же из кэша МПК ../uzsign_ai_v2/cache/uzsl.npz (из него
собран cache/M13/uzsl.npz, m13_build.tasks_for) + зеркальность позы и кистей (elbow_check, к3 0047 разд. 2).
→ cache/UZSOHN/alluz_pose.npz (кадры = cache/M13/alluz.npz, до filter_nans) и cache/UZSOHN/alluz_x10_pose.npz (кадры =
uzsign_ai_v2/datasets_local/UZSL_X10/uzsl_x10_sohn.npz). Персональные данные — только ПК.
Источник позы — кэш МПК ALL-UZ (85 точек; 0–3 = поза 11–14 из Э0; 0 = нет точки), кадр Sohn → кадр МПК по m13_build.to30
(как в alluz_x10.synth_record). Синтетика ×10: тот же синтез (перенос длин костей + амплитуда + растяжение, затравки и люди
те же), поза берётся из того же результата 85 точек (плечи и локти перенесены bonesynth, локоть — та же амплитуда). Проверка:
хэш полной записи Sohn ×10 (118 точек), посчитанной здесь, = хэш записи в кэше ×10 (значит, кадры и синтез совпали).
"""
import hashlib
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(UZ / "uzsign_ai_v2/scripts"))
OUT = ROOT / "cache/UZSOHN"
NP = 4      # первые 4 точки МПК: POSE_MP (11, 12, 13, 14)
G: dict = {}


def pose_of(x85: np.ndarray, pres: np.ndarray) -> np.ndarray:
    p = x85[:, :NP].astype(np.float32).copy()
    bad = ~pres[:, :1] | (p == 0).all(-1)
    p[bad] = np.nan
    return p


def real() -> tuple[list, list, dict, dict]:
    from mpk import data
    import alluz
    import m13_build
    A = data.load(alluz.CACHE)
    S = m13_build.load("alluz")
    assert [c["sample_id"] for c in A["clips"]] == [c["sample_id"] for c in S["clips"]]
    assert data.POSE_MP[:NP] == (11, 12, 13, 14)
    out = []
    for i in range(len(A["clips"])):
        a0, a1 = A["offsets"][i], A["offsets"][i + 1]
        f = m13_build.to30(a1 - a0, float(S["fps_src"][i]))
        assert len(f) == S["offsets"][i + 1] - S["offsets"][i], i
        out.append(pose_of(A["xy"][a0:a1][f], A["present"][a0:a1][f]))
    return out, A["clips"], A, S


def synth_pose(i: int) -> list:
    """= alluz_x10.synth_record(i) (строка в строку для 118 точек), но выдаёт позу 4 точек и хэш записи Sohn."""
    from mpk import bonesynth as bs
    import alluz_x10 as X
    import m13_build
    A, S, cl = G["A"], G["S"], G["clips"][i]
    a0, a1_ = A["offsets"][i], A["offsets"][i + 1]
    xy = A["xy"][a0:a1_].astype(np.float32)
    pres = A["present"][a0:a1_]
    asp = np.full(len(xy), float(A["aspect"][i]))
    s0, s1 = S["offsets"][i], S["offsets"][i + 1]
    sx = S["xy"][s0:s1]
    fidx = m13_build.to30(len(xy), float(S["fps_src"][i]))
    out = []
    for j in range(1, X.K + 1):
        r, _ = bs.ratios(G["people"][j - 1], G["own"][cl["signer_id"]])
        new, _ = bs.retarget(xy, asp, r)
        rng = np.random.default_rng([X.SEED, X.h32(cl["sample_id"]), j])
        s = float(np.exp(rng.uniform(np.log(X.STRETCH[0]), np.log(X.STRETCH[1]))))
        amp = float(rng.uniform(*X.AMP))
        new = X.amplitude(new, amp)
        o85, n85 = xy[fidx], new[fidx]
        sn = sx.copy()
        for lo, mp in ((G["SL"], 6), (G["SR"], 27)):
            sn[:, lo] += n85[:, mp:mp + 21] - o85[:, mp:mp + 21]
        sn[:, G["SF"]] = n85[:, None, 49] + r[G["NEAR"]][None, :, None] * (sx[:, G["SF"]] - o85[:, None, 49])
        is_ = X.stretch_idx(len(sx), s)
        if np.isnan(sn[is_]).all():
            is_ = np.arange(len(sx))
        so = sn[is_]
        keep = ~np.isnan(so).all((1, 2))
        out.append((pose_of(n85[is_][keep], pres[fidx][is_][keep]), hashlib.sha1(np.ascontiguousarray(so[keep]).tobytes()).hexdigest()))
    return out


def save(name: str, poses: list, ref_offsets: np.ndarray, extra: dict) -> None:
    lens = np.array([len(p) for p in poses])
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    assert np.array_equal(off, ref_offsets), f"{name}: кадры позы ≠ кадры кэша Sohn"
    xy = np.concatenate(poses).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    dst = OUT / f"{name}_pose.npz"
    assert not dst.exists(), f"{dst} уже есть (не перезаписывается)"
    info = {"points_mediapipe_pose": [11, 12, 13, 14], "frames": int(len(xy)), "nan_frac": float(np.isnan(xy[..., 0]).mean()), **extra}
    np.savez(dst, xy=xy, offsets=off, info=np.array(json.dumps(info, ensure_ascii=False)))
    print(f"{dst}: {json.dumps(info, ensure_ascii=False)}", flush=True)


def build_x10(poses_real: list, clips: list, A: dict, S: dict) -> None:
    from mpk import bonesynth as bs, data, sohn
    import alluz_x10 as X
    import m12_3_prepare
    t0 = time.time()
    signers = sorted({c["signer_id"] for c in clips})
    asp = m12_3_prepare.per_frame_aspect(A)
    own = {}
    for s in signers:
        r = m12_3_prepare.rows_of(A, [i for i, c in enumerate(clips) if c["signer_id"] == s])
        own[s] = bs.measure(A["xy"][r].astype(np.float32), asp[r])
    rel_s = m12_3_prepare.measure_signers(data.load(X.OM / "cache/slovo.npz"))
    full = sorted(s for s, v in rel_s.items() if np.isfinite(v).all())
    donors = [str(x) for x in np.random.default_rng(X.SEED).choice(full, X.K, replace=False)]
    man = json.loads((X.DS / "manifest.json").read_text())
    assert donors == man["donors_slovo"], (donors, man["donors_slovo"])
    idx30 = [i for i in range(len(clips)) if abs(float(S["fps_src"][i]) - 30) <= 1][::10]
    near = X.face_near(A, S, idx30)
    assert {int(k): int(m) for k, m in zip([k for k in sohn.POINT_LANDMARKS if k < 468], near)} == {int(k): v for k, v in man["face_near"].items()}
    loc = lambda ks: np.array([sohn.LOCAL[k] for k in ks])
    G.update(A=A, S=S, clips=clips, own=own, people=[rel_s[d] for d in donors], NEAR=near,
             SL=loc(sohn.LHAND), SR=loc(sohn.RHAND), SF=loc([k for k in sohn.POINT_LANDMARKS if k < 468]))
    with get_context("fork").Pool(12) as pool:
        res = pool.map(synth_pose, range(len(clips)), chunksize=16)
    print(f"синтез позы: {time.time() - t0:.0f} с", flush=True)
    keep_real = []
    for i in range(len(clips)):
        x = S["xy"][S["offsets"][i]:S["offsets"][i + 1]]
        keep_real.append(poses_real[i][~np.isnan(x).all((1, 2))])
    with np.load(X.SOHN_NPZ, allow_pickle=False) as z:
        off = z["offsets"]
        xy = z["xy"]
    n = len(clips)
    k = n
    for i, outs in enumerate(res):          # порядок кэша ×10: реальные, затем запись 1 × люди 1–10, запись 2 × …
        for p, h in outs:
            got = hashlib.sha1(np.ascontiguousarray(xy[off[k]:off[k + 1]]).tobytes()).hexdigest()
            assert got == h, f"запись ×10 {k}: синтез не совпал с кэшем"
            k += 1
    assert k == len(off) - 1
    del xy
    save("alluz_x10", keep_real + [p for outs in res for p, _ in outs], off,
         {"source": "uzsl_x10_sohn.npz", "x10_hash_check": f"{k - n} синтетических записей совпали побайтно"})


def elbow_check(pose: np.ndarray, sx: np.ndarray, off: np.ndarray) -> dict:
    """к3 0047 разд. 2 (б): поза и кисти отражены одинаково (иначе U01a поставит левый локоть к правой кисти). Кадры с обеими
    кистями и локтем; у записи — медиана по кадрам, затем медиана по записям: ‖локоть − запястье своей кисти‖ < ‖… чужой‖."""
    from mpk import sohn
    wl, wr = sx[:, sohn.LH_LOC[0]], sx[:, sohn.RH_LOC[0]]
    both = np.isfinite(wl[:, 0]) & np.isfinite(wr[:, 0])
    rec = np.repeat(np.arange(len(off) - 1), np.diff(off))
    out = {}
    for side, el, w_own, w_other in (("L", pose[:, 2], wl, wr), ("R", pose[:, 3], wr, wl)):   # точки позы 2, 3 = MediaPipe 13, 14
        m = both & np.isfinite(el[:, 0])
        d = np.stack([np.linalg.norm(el - w_own, axis=1), np.linalg.norm(el - w_other, axis=1)], 1)[m]
        per = [np.median(c, 0) for c in np.split(d, np.flatnonzero(np.diff(rec[m])) + 1)] if m.any() else []
        assert per, f"локоть {side}: нет кадров с обеими кистями"
        own, other = np.median(per, 0)
        out[side] = {"records": len(per), "own": round(float(own), 4), "other": round(float(other), 4)}
        assert own < other, f"зеркальность позы и кистей: локоть {side} ближе к чужой кисти ({own:.4f} ≥ {other:.4f})"
    return out


def real_sb(name: str) -> None:
    from mpk import data, sohn
    import m13_build
    src = "../uzsign_ai_v2/cache/uzsl.npz" if name == "uzsl" else f"cache/{name}.npz"   # кэш МПК, из которого собран cache/M13
    A = data.load(ROOT / src)
    S = m13_build.load(name)
    assert [c["sample_id"] for c in A["clips"]] == [c["sample_id"] for c in S["clips"]]
    assert data.POSE_MP[:NP] == (11, 12, 13, 14) and (data.MPK_POINTS[6], data.MPK_POINTS[27]) == (33, 54)   # 6 — запястье Л, 27 — П
    out, worst = [], 0.0
    for i in range(len(A["clips"])):
        a0, a1 = A["offsets"][i] + A["lo"][i], A["offsets"][i] + A["hi"][i]
        f = m13_build.to30(a1 - a0, float(S["fps_src"][i]))
        assert len(f) == S["offsets"][i + 1] - S["offsets"][i], i
        xy, pr = A["xy"][a0:a1][f].astype(np.float32), A["present"][a0:a1][f]
        sx = S["xy"][S["offsets"][i]:S["offsets"][i + 1]]
        for mp, loc, part in ((6, sohn.LH_LOC[0], 1), (27, sohn.RH_LOC[0], 2)):
            m = pr[:, part] & np.isfinite(sx[:, loc, 0])
            if m.any():
                worst = max(worst, float(np.abs(xy[m, mp] - sx[m, loc]).max()))
        out.append(pose_of(xy, pr))
    assert worst < 2e-3, f"{name}: кадры МПК ≠ кадрам Sohn (запястья, {worst})"
    extra = {"elbow_check": elbow_check(np.concatenate(out), S["xy"], S["offsets"])} if name == "uzsl" else {}
    save(name, out, S["offsets"], {"source": f"{src} (МПК, [lo, hi)) → cache/M13/{name}.npz", "wrist_check_max_abs": worst, **extra})


def main() -> None:
    names = sys.argv[1:] or ["alluz", "alluz_x10"]
    for n in ("slovo", "bukva", "uzsl"):
        if n in names:
            real_sb(n)
    names = [n for n in names if n not in ("slovo", "bukva", "uzsl")]
    if not names:
        return
    poses, clips, A, S = real()
    if "alluz" in names:
        save("alluz", poses, S["offsets"], {"source": "cache/M13/alluz.npz (до filter_nans)"})
    if "alluz_x10" in names:
        build_x10(poses, clips, A, S)


if __name__ == "__main__":
    main()
