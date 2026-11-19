"""UzSohn, этап 1 — отбор идей (reports/UZSOHN_prereg.md; черновик UZSOHN_prereg_DRAFT.md сохраняется).

cd osnovnaya_model && PYTHONPATH=src:../uzsign_e1/src:../uzsign_ai_v2/src python scripts/uzsohn_sel.py <команда>
  base                 база по правилу разд. 4: Sohn ×10 (alluz_x10), только если H2 (Sohn ×10 − Sohn) установлена в плюс
                       по 99 % ДИ (reports/ALLUZ_X10_results.json), иначе Sohn ALL-UZ (alluz) → runs/UZSOHN/base.json, печать набора
  ideas FILE           порядок шагов этапа 1 из файла (U02 U01a U04 через пробел/строки, # — комментарий) → печать
  decide CUR CAND IDEA отбор (разд. 3): CAND принимается, если macro по 10 V_k − macro CUR ≥ max(1; δ) п. п. и CAND лучше
                       у ≥ 6 из 10 V_k → runs/UZSOHN/decide_<CAND>.json; код выхода 0 — принята, 10 — не принята
  report TAG…          таблица этапа 1 → reports/UZSOHN_stage1.md (параметры, из них stem, мс/запись ЦП, часы ГП, примеры; решения
                       шагов и ворот U04 без V_j = S_k (ворота U01a — только печать, q без V_j = S_k — справка), условие U01b,
                       Δ_k S11/S12 у U01a, статистики U02 и доля ветки Sohn опоры U02, код прогонов, оценка ложного принятия)
  report2 CUR CAND     этап 2 (U01b, UZSOHN_prereg_popravka2.md): CAND против контроля этапа 1 CUR → reports/UZSOHN_stage2.md
  e2 CUR CAND STEP     Э2 турнира (поправка 4 к reports/TURNIR_prereg.md, 22.3–22.7, 22.10 п. 2): шаг STEP (U02-1 U02-2 U01a U04
                       U01b r2 r3 r4) — CAND против текущей CUR (суффиксы тегов uzsohn_e2_<набор>_…: v0, in_<рычаги через ->)
                       на 10 ячейках валидации Slovo и Bukva, правило judge по micro top-1 → runs/UZSOHN/decide_e2_<STEP>.json
                       (не перезаписывается); код выхода 0 — принят, 10 — не принят
Счёт — только metrics.json прогонов runs/UzSohn/<тег>/<фолд>/seed48 (top1 на V_k); тест S_k не читается. e2 — только выходы *_val
прогонов --val-only (тест Slovo и Bukva они не считают) и log.csv M13 (валидация).
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UZ = ROOT.parent
RUNS = ROOT / "runs/UzSohn"
OUT = ROOT / "runs/UZSOHN"
CODES = UZ / "uzsign_ai_v2/data/signer_codes.local.json"   # только ПК; в итог — только коды S..
DELTA = 1.0081957667939179     # наибольшая разница macro двух из трёх затравок Sohn ALL-UZ (42/44; разд. 3)
THR = max(1.0, DELTA)
MIN_BETTER = 6
SEED = 48                      # отбор целиком на 48: контроль sb/all/seed48, предобучения плеч, дообучение (разд. 5; к1 0123 п. 20)
EPOCHS, PRE_STEPS = 300, 13_800   # рецепт Sohn (sohn.CFG["epoch"], разд. 8); шаги каждого предобучения S+B (разд. 5, 7)
IDEA = {1: "parts", 2: "hand_gcn", 3: "pose", 4: "hand_cross", 5: "ms_conv", 6: "attn_pool", 7: "cos_head"}
FOLDS = sorted(json.loads((UZ / "uzsign_ai_v2/splits/uzsl_allz_loso.json").read_text())["folds"])


def pre_of(inp) -> str:
    """Предобучение отбора для входа (разд. 5, п. 20): пустой — контроль sb/all/seed48, иначе sb_in_<токены через ->/all/seed48."""
    return f"{'sb_in_' + '-'.join(inp) if inp else 'sb'}/all/seed{SEED}/best.pt"


def full_budget(m: dict) -> bool:
    """Полный бюджет рецепта (разд. 7: равное число примеров): без --limit-train, --stop-epoch, --max-steps и --epochs ≠ 300.
    Цепочка пропускает прогон, если есть metrics.json, — урезанная проба в путь этапа иначе прошла бы молча (раунд 3 проверки р4)."""
    return (m.get("steps_run") == m.get("steps") and m.get("limit_train") == 0 and m.get("epochs") == EPOCHS
            and not m.get("stop_epoch") and not m.get("max_steps"))


def scores(tag: str) -> dict[str, float]:
    out = {}
    for f in FOLDS:
        m = json.loads((RUNS / tag / f / f"seed{SEED}/metrics.json").read_text())
        assert m["inner_val"] and m["inner_val"] != f and m["mode"] == "all", (tag, f)
        assert full_budget(m), (tag, f, "бюджет", *(m.get(k) for k in ("steps_run", "steps", "limit_train", "epochs", "stop_epoch")))
        ini = (m.get("init") or {}).get("init_from") or ""   # п. 20: дообучение 48 ← предобучение 48 своего входа (не sb/all/seed42)
        assert m["seed"] == SEED and ini.split("runs/UzSohn/")[-1] == pre_of(m.get("input")), (tag, f, m.get("seed"), ini)
        out[f] = 100 * m["top1"]
    assert len(out) == 10
    return out


def base() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r = json.loads((UZ / "uzsign_ai_v2/reports/ALLUZ_X10_results.json").read_text())
    h2 = r["H2_Sohnx10_minus_Sohn_99"]
    assert h2["level"] == 0.99
    s = "alluz_x10" if h2["ci"][0] > 0 else "alluz"
    d = {"set": s, "H2": h2, "H2_verdict": r["H2_verdict"], "rule": "×10 только при нижней границе 99 % ДИ H2 > 0",
         "time": time.strftime("%F %T")}
    p = OUT / "base.json"
    if p.is_file():
        old = json.loads(p.read_text())
        assert old["set"] == s, f"база уже выбрана иначе: {old['set']} ≠ {s}"
    else:
        p.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print(s)


STEPS = ("U02", "U01a", "U04")   # UZSOHN_prereg разд. 3: согласованный порядок этапа 1 (к1 0111, к3 0018, к2 0133)


def ideas(path: str) -> None:
    v = [t for line in Path(path).read_text().splitlines() for t in line.split("#")[0].split()]
    assert tuple(v) == STEPS, f"шаги {v}: этап 1 — ровно {' '.join(STEPS)} (UZSOHN_prereg)"
    print(" ".join(v))


def judge(a: dict, b: dict) -> tuple[dict, float, int, bool]:
    d = {f: b[f] - a[f] for f in a}
    diff = float(np.mean(list(d.values())))
    better = sum(x > 0 for x in d.values())
    return d, diff, better, diff >= THR and better >= MIN_BETTER


def ok_delta(d: dict) -> bool:
    """Правило decide по готовым Δ_k: Δ macro ≥ THR и Δ_k > 0 у ≥ MIN_BETTER (для 9 V_j — тоже ≥ 6)."""
    return float(np.mean(list(d.values()))) >= THR and sum(x > 0 for x in d.values()) >= MIN_BETTER


def nested(per: dict, vk_of: dict, rule) -> dict:
    """Ревизия р2 (UZSOHN_prereg разд. 2, угроза 11): записи S_k участвуют в отборе как V соседнего фолда. Для каждого внешнего
    k — то же правило без фолда j, у которого V_j = S_k (по 9 V_j ≠ S_k) → {k: решение}. Только печать: решение этапа — общее.
    S_k при этом остаётся в обучении 8 из 9 оставшихся фолдов (и в их прототипах и классах ворот): это не вложенный отбор."""
    return {k: rule({j: v for j, v in per.items() if vk_of[j] != k}) for k in per}


def nested_line(name: str, overall, per: dict, vk_of: dict, rule, codes: dict, val=None, label: str = "") -> str:
    r = nested(per, vk_of, rule)
    bad = [codes.get(k, "?") for k in sorted(r) if r[k] != overall]
    v = [x for x in nested(per, vk_of, val).values() if x is not None] if val else []
    return (f"{name}: решение без V_j = S_k (по 9 V_j ≠ S_k) совпало с общим у {len(r) - len(bad)}/{len(r)} внешних k"
            + (f"; {label} без V_j = S_k — от {min(v):+.3f} до {max(v):+.3f}" if v else "")
            + (f"; не совпало у {', '.join(sorted(bad))} — подтверждение на этих S_k не вложенное" if bad else ""))


def u01b_note(u04_gate: str | None, u04_accepted: bool | None) -> str:
    """Условие к1 0098 (В3) и к3 0011: U01b отложена, «вернуться к U01b, только если U04 не пройдёт порог»."""
    head = "U01b (отложена; к1 0098: «вернуться к U01b, только если U04 не пройдёт порог», к3 0011): "
    if u04_gate == "skip":
        return head + ("ворота U04 закрыты — U04 не проверена (при +5 п. п. на буквах ворота (ii) открываются лишь в 0,16–0,58, "
                       "разд. 4.3), выполнено ли условие, не определено; решение после этапа 1 за автором")
    if u04_accepted is None:
        return head + "U04 не дошла до решения — решение после этапа 1 за автором"
    return head + ("U04 принята — условие возврата не наступило" if u04_accepted else
                   "U04 не прошла порог — условие выполнено: U01b — кандидат отдельным плечом после этапа 1 (решение автора)")


def code_note(ms: list[dict]) -> str:
    """Ревизия р2–р3: на каком коде шли прогоны этапа (commit, dirty из metrics.json m13_train — до обучения); код коммитов
    сверяется git diff по файлам этапа (scripts/uzsohn_code.txt — тот же список, что у цепочки и m13_train.git)."""
    import m13_train
    om = m13_train.uzsohn_code()[0]
    g = [m.get("git_osnovnaya_model") or {} for m in ms]
    cm = sorted({x.get("commit") or "?" for x in g})
    dirty = sum(bool(x.get("dirty")) for x in g)
    same = all(subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", cm[0], c, "--", *om]).returncode == 0
               for c in cm[1:]) if "?" not in cm else False
    warn = dirty > 0 or not same
    return (("**ВНИМАНИЕ:** " if warn else "") + f"код прогонов этапа ({len(ms)}): коммиты {', '.join(c[:7] for c in cm)}; код "
            f"этапа (scripts/uzsohn_code.txt) между ними {'одинаков' if same else 'РАЗЛИЧАЕТСЯ или неизвестен'}; с незакоммиченным "
            f"кодом этапа (dirty) — {dirty}")


FALSE_ACCEPT = ("Ошибка жадного отбора (оценка по модели шума затравок базы, ревизия р2; разд. 11, угроза 1): при нулевом эффекте "
                "шаг принимается с вероятностью ≈ 0,05–0,11, хотя бы один ложный шаг за этап (3–4 шага: U02-1, U02-2 и U01a всегда, "
                "U04 — по воротам) — ≈ 0,13–0,28; принятое плечо несёт свой удачный шум (+0,3…+0,8 п. п.), и следующий шаг "
                "сравнивается с теми же прогонами — «не принята» после принятого шага читать с этой поправкой (шаг с истинным +1 п. п. "
                "проходит в ≈ 0,14–0,35 против ≈ 0,5). Модель считает шум кандидата и текущей независимым, а у контроля и плеч общие "
                "потоки случайных чисел затравки 48 — при положительной связи шума оценки скорее верхние (угроза 1).")


RISK2 = ("Риск отбора (модель шума затравок базы; угроза 1 предрегистрации, Поправка 2 разд. 3.5): одного шага — при нулевом эффекте "
         "шаг принимается с вероятностью ≈ 0,05–0,11; накопленный по серии — хотя бы один ложный шаг из 5 плеч этапов 1–2 (все против "
         "одного контроля) — ≈ 0,19–0,32. Модель считает шум плеч и контроля независимым; у них общие потоки случайных чисел затравки "
         "48 — при положительной связи оценки скорее верхние (угроза 1).")   # этап 2: дословно Поправка 2, 3.5


def code_vs(ctrl: list[dict], cand: list[dict]) -> str:
    """Этап 2 (Поправка 2, 3.4): код этапа прогонов U01b против кода прогонов контроля — одной строкой (code_note сверяет код
    только внутри одного плеча)."""
    import m13_train
    om = m13_train.uzsohn_code()[0]
    cm = lambda ms: sorted({(m.get("git_osnovnaya_model") or {}).get("commit") or "?" for m in ms})
    a, b = cm(ctrl), cm(cand)
    same = "?" not in a + b and all(subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", x, y, "--", *om]).returncode == 0
                                    for x in a for y in b)
    return (f"Код этапа (scripts/uzsohn_code.txt) U01b ({', '.join(c[:7] for c in b)}) против контроля ({', '.join(c[:7] for c in a)}): "
            f"{'одинаков' if same else 'РАЗЛИЧАЕТСЯ или неизвестен'} — ожидаемо «РАЗЛИЧАЕТСЯ»: U01b добавлена в код этапа только "
            "добавлением; путь Sohn — бит в бит с базой (тест), сравнение с готовым контролем — Поправка 2, 3.4 пп. 1–3")


def selftest() -> None:
    a = {str(i): 30.0 for i in range(10)}
    assert not judge(a, {k: v + 1.0 for k, v in a.items()})[3]                 # +1,00 < δ = 1,008
    assert judge(a, {k: v + 1.01 for k, v in a.items()})[3]
    up6 = {k: v + (3.0 if int(k) < 6 else -1.0) for k, v in a.items()}         # +1,4, лучше у 6
    assert judge(a, up6)[3] and judge(a, up6)[2] == 6
    up5 = {k: v + (5.0 if int(k) < 5 else -1.0) for k, v in a.items()}         # +2,0, но лучше у 5
    assert not judge(a, up5)[3]
    sv = {t: {"q": q, "decision": d} for t, q, d in (("40", 0.6, "run"), ("50", 0.4, "skip"), ("60", None, "skip"))}
    g1 = {"sensitivity": sv, "stable_40_60": False, "decision": "skip", "n": 25, "q_ci99": [0.16, 0.68]}
    assert "неустойчиво" in gate_note("u01a", g1) and "n = 25" in gate_note("u01a", g1) and "[0.160; 0.680]" in gate_note("u01a", g1)
    assert "неустойчиво" not in gate_note("u01a", {**g1, "stable_40_60": True})
    for dd in ("skip", "run"):   # (в), 12 п. 5: ворота U01a — только печать при любом решении правила (к1 0125, к3 0033)
        s = gate_note("u01a", {**g1, "decision": dd})
        assert "вырождены" in s and "только печать" in s and "не проверена" not in s and "решение" not in s
    pf = {str(i): {"vk": str((i + 1) % 10), "place_50": 4, "only_shoulders_50": int(i == 0)} for i in range(10)}   # q = 1/40
    s = gate_note("u01a", {**g1, "per_fold": pf})                            # без фолда «0» (V_0 = S_1) q = 0, без прочих — 1/36
    assert "q без V_j = S_k (по 9 V_j ≠ S_k; справка) — от 0.000 до 0.028" in s and "не вложенное" not in s
    cv = {"frame_median": 0.13, "frame_mean": 0.12, "pixel_median": 0.02, "pixel_mean": 0.03}
    st = {"cv_sy_sx": cv, "cv_face_width_u02": {"median": 0.05, "mean": 0.06}, "pixel_iqr_over_median_within_signer_median": None,
          "pixel_iqr_over_median_all_records": 0.5, "abs_ln_median_pixel": 0.9, "abs_ln_median_frame": 0.3}
    assert "ворот нет" in u02_note(st) and "pixel_mean 0.030" in u02_note(st) and "в пикселях 0.900" in u02_note(st)
    assert "в долях кадра 0.300" in u02_note(st) and "в долях кадра —" in u02_note({**st, "abs_ln_median_frame": None})
    assert "медиана своего исполнителя) 0.500" in u02_note(st) and "по исполнителям —" in u02_note(st)
    g4 = {"decision": "skip", "diff_harf": 2.04, "ci99_harf": [-1.7, 6.1], "n_harf": 294, "harf_share_S11_S12": 257 / 294,
          "diff_all_ref": 0.3, "palm_frames_lt_half_median": 0.0307, "one_plus_cos_zero_visible": 0.0008, "level": 0.99}
    assert "не проверена" in gate_note("u04", g4) and "87.4 %" in gate_note("u04", g4) and "[-1.70; +6.10]" in gate_note("u04", g4)
    assert "99 % ДИ" in gate_note("u04", g4) and "90 % ДИ" in gate_note("u04", {**g4, "level": 0.9})   # уровень — из gate_u04.json
    assert "3.07 %" in gate_note("u04", g4) and "0.080 %" in gate_note("u04", g4) and "страт" in gate_note("u04", g4)
    b = {"train_examples": 1000, "train_sohn_branch": 62, "eval_records": 50, "eval_sohn_branch": 0}
    bn = branch_note([{"set": "sb", "u02_sohn_branch": b}, {"set": "alluz", "u02_sohn_branch": b}, {"set": "alluz", "u02_sohn_branch": None}])
    assert "alluz: обучение 62/1000 (6.20 %), оценка 0/50" in bn and "sb:" in bn and branch_note([{"set": "sb"}]) == ""
    assert "без U02" in bn and "ВНИМАНИЕ" not in bn
    u = {"set": "alluz", "input": ["u02s", "u01a"], "steps_run": 2, "cfg": {"batch_size": 500}, "tag": "t", "fold": "f1"}
    assert "ВНИМАНИЕ" not in branch_note([{**u, "u02_sohn_branch": b}])                  # 1000 = 2 × 500
    assert "ВНИМАНИЕ" in branch_note([{**u, "u02_sohn_branch": b, "steps_run": 3}]) and "t/f1" in branch_note([{**u, "u02_sohn_branch": None}])
    assert "ВНИМАНИЕ" in branch_note([{**u, "u02_sohn_branch": None}, {"set": "sb", "u02_sohn_branch": b}])
    r4 = gate_note("u04", {**g4, "decision": "run"})                         # метод ДИ (ii) — и при открытых воротах
    assert "не проверена" not in r4 and "страт" in r4 and "иерарх" not in r4 and "нет" in gate_note("u04", {"decision": "skip"})
    vk = {str(i): str((i + 1) % 10) for i in range(10)}                       # V фолда i — следующий по кругу
    d = {str(i): 2.0 for i in range(10)}
    assert all(nested(d, vk, ok_delta).values())                                # +2 у всех: без любого S_k — то же
    d6 = {k: (3.0 if int(k) < 6 else -1.0) for k in d}                         # общее: +1,4, лучше у 6/10 → принята
    assert ok_delta(d6) and sum(not v for v in nested(d6, vk, ok_delta).values()) == 6   # без любого из 6 «лучших» V — 5 из 9
    assert "не совпало у" in nested_line("x", True, d6, vk, ok_delta, {k: f"S{k}" for k in d})
    assert "не определено" in u01b_note("skip", None) and "условие выполнено" in u01b_note("run", False)
    assert "не наступило" in u01b_note("run", True)
    print("selftest ok")


def decide(cur: str, cand: str, idea: str) -> None:
    a, b = scores(cur), scores(cand)
    d, diff, better, ok = judge(a, b)
    res = {"idea": idea, "current": cur, "candidate": cand, "macro_current": round(float(np.mean(list(a.values()))), 3),
           "macro_candidate": round(float(np.mean(list(b.values()))), 3), "diff": round(diff, 3), "better": f"{better}/10",
           "threshold": round(THR, 4), "delta": DELTA, "accepted": ok, "per_vk": {f: round(x, 2) for f, x in d.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"decide_{cand}.json"
    if p.is_file():   # повтор после перезапуска — то же решение; файл не перезаписывается (копии вместо перезаписи)
        old = json.loads(p.read_text())
        assert old["accepted"] == ok and old["per_vk"] == res["per_vk"], f"{p}: решение изменилось"
    else:
        p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("idea", "macro_current", "macro_candidate", "diff", "better", "accepted")}, ensure_ascii=False))
    sys.exit(0 if ok else 10)


def ms_per_record(flags: dict, n_classes: int, T: int, n: int = 30, tokens=()) -> float:
    """мс на запись, ЦП, 1 поток, пакет 1: preprocess + прямой проход (вход — случайные точки длины T)."""
    import torch
    from mpk import sohn
    torch.set_num_threads(1)
    torch.manual_seed(0)
    m = sohn.SohnNet(n_classes, channels=sohn.input_channels(tokens), tokens=tokens, **flags).eval()
    rng = np.random.default_rng(0)
    P = sohn.NUM_NODES_POSE if flags.get("pose") or tokens else sohn.NUM_NODES
    pre0 = (lambda x: sohn.preprocess_u(x, 0.5625, tokens)) if tokens else sohn.preprocess_pose if flags.get("pose") else sohn.preprocess
    # № 2 Э2: маска частей — в конце входа, как AugSet m13_train
    pre = (lambda x: np.concatenate([pre0(x), sohn.part_mask(x, tokens)], 1)) if flags.get("part_tokens") else pre0
    xs = [rng.uniform(0.2, 0.8, (T, P, 2)).astype(np.float32) for _ in range(n + 5)]
    ts = []
    with torch.no_grad():
        for i, x in enumerate(xs):
            t0 = time.perf_counter()
            m(*sohn.collate([pre(x)], pad_to=1))
            if i >= 5:
                ts.append(time.perf_counter() - t0)
    return round(1000 * float(np.median(ts)), 2)


def params_stem(m: dict) -> int:
    """Параметры входа сети по metrics.json прогона: stem — первый Dense без смещения (к1 0122 п. 3); у № 2 Э2 stem снят, вместо
    него — токены частей tok (поправка 4, 22.11; 0163 разд. 6)."""
    if (m.get("flags") or {}).get("part_tokens"):
        from mpk import sohn
        return sum(p.numel() for p in sohn.PartTokens(tuple(m["input"]), m["cfg"]["dim"]).parameters())
    return m.get("channels", 708) * m["cfg"]["dim"]


def u02_note(gj: dict) -> str:
    """Строка итога к статистикам U02 (stats_u02.json; разд. 4.1): ворот нет — только печать (к1 0124 п. 2, к3 0032 п. 2)."""
    wi, al = gj.get("pixel_iqr_over_median_within_signer_median"), gj.get("pixel_iqr_over_median_all_records")
    fr = gj.get("abs_ln_median_frame")
    return ("U02 — ворот нет: два плеча U02-1 → U02-2 и медиана заранее (к1 0124 п. 2, к3 0032 п. 2); справочно: КВ s_y/s_x между "
            f"исполнителями {', '.join(f'{k} {v:.3f}' for k, v in gj['cv_sy_sx'].items())}; IQR / медиана пиксельного s_y/s_x от записи "
            f"к записи — медиана по исполнителям {'—' if wi is None else format(wi, '.3f')}, по всем записям (r / медиана своего "
            f"исполнителя) {'—' if al is None else format(al, '.3f')}; |ln медианы s_y/s_x| в пикселях {gj['abs_ln_median_pixel']:.3f}, "
            f"в долях кадра {'—' if fr is None else format(fr, '.3f')} (справочно: механизм U02-1 есть всегда, и ни один уровень "
            "его размер не задаёт — разд. 4.1, угроза 12); КВ ширины лица после U02: "
            + ", ".join(f"{k} {v:.3f}" for k, v in gj["cv_face_width_u02"].items()))


def branch_note(ms: list[dict]) -> str:
    """Опора U02 после масок, вариант (в) (разд. 4.1; к1 0124 п. 3, к3 0032 п. 3): доля примеров без кадра с обоими плечами —
    ветка Sohn (нормировка Sohn; вход = вход плеча без U02 — у U02-1 и U02-2 Sohn бит в бит) при обучении (S+B, ALL-UZ) и на
    оценке (тест V_k; val S+B) — из metrics.json. Прогон с u02 без счётчика или с train_examples ≠ steps_run × пакет —
    «ВНИМАНИЕ» (печать, которую просили консультанты, не пропадает молча)."""
    agg, bad = {}, []
    for m in ms:
        b = m.get("u02_sohn_branch")
        if {"u02s", "u02c"} & set(m.get("input") or ()) and (
                not b or b["train_examples"] != m.get("steps_run", -1) * (m.get("cfg") or {}).get("batch_size", 0)):
            bad.append(f"{m.get('tag')}/{m.get('fold')}")
        if b:
            a = agg.setdefault(m["set"], [0, 0, 0, 0])
            for j, k in enumerate(("train_sohn_branch", "train_examples", "eval_sohn_branch", "eval_records")):
                a[j] += b[k]
    warn = (f"**ВНИМАНИЕ:** у {len(bad)} прогонов с u02 нет доли ветки Sohn или train_examples ≠ steps_run × пакет: "
            + ", ".join(bad[:5]) + (" …" if len(bad) > 5 else "")) if bad else ""
    if not agg:
        return warn
    return ("Опора U02 — после масок (вариант (в)): ветка Sohn (нет кадра с обоими плечами; нормировка Sohn, вход = вход плеча без "
            "U02 — у U02-1 и U02-2 Sohn бит в бит) — "
            + "; ".join(f"{s}: обучение {a[0]}/{a[1]} ({100 * a[0] / max(a[1], 1):.2f} %), оценка {a[2]}/{a[3]}"
                        for s, a in sorted(agg.items()))
            + ". Там U02 не действует; знак влияния ветки Sohn на Δ U02 не установлен (BN части, угроза 14; к3 0033 разд. 2, "
              "к1 0126 разд. Е): у u02s эти примеры — выбросы масштаба и через BN части меняют нормировку остальных"
            + (f". {warn}" if warn else ""))


def gate_note(g: str, gj: dict) -> str:
    """Строка итога к воротам (UZSOHN_prereg разд. 4.2–4.3, 12): U01a — только печать (вариант (в), 12 п. 5): q при 40/50/60,
    «неустойчиво», q без V_j = S_k (справка, «не вложенное» не помечается — ворота ничего не решают); U04 — оговорка о
    чувствительности ворот, печать (ладонь, 1 + cos)."""
    if g == "u01a":
        import uzsohn_gates as G
        sv = gj["sensitivity"]
        q = lambda v: "—" if v["q"] is None else f"{v['q']:.3f}"
        ci = gj.get("q_ci99")
        pf = gj.get("per_fold") or {}
        q9 = lambda s: G.u01a_rule(sum(r[f"place_{G.PCT}"] for r in s.values()), sum(r[f"only_shoulders_{G.PCT}"] for r in s.values()))[0]
        qj = [x for x in nested(pf, {f: r["vk"] for f, r in pf.items()}, q9).values() if x is not None]
        return (f"q при порогах 40 / 50 / 60: {' / '.join(q(sv[t]) for t in ('40', '50', '60'))} (по правилу q ≥ {G.Q_MIN}, только "
                f"печать: {' / '.join(sv[t]['decision'] for t in ('40', '50', '60'))}); при 50: n = {gj.get('n')} ошибок «место», 99 % ДИ q "
                + ("—" if ci is None else f"[{ci[0]:.3f}; {ci[1]:.3f}]") + " (иерархический бутстреп по V_k, затем ошибкам; справочно)"
                + ("" if gj["stable_40_60"] else "; **неустойчиво**: при 40 или 60 правило дало бы иное (разд. 4.2)")
                + (f"; q без V_j = S_k (по 9 V_j ≠ S_k; справка) — от {min(qj):.3f} до {max(qj):.3f}" if qj else "")
                + "; ворота вырождены (разд. 4.2; ошибка определения к3 0015) — только печать: плечо U01a поставлено при любом q "
                  "(вариант (в), разд. 12 п. 5; к1 0125, к3 0033)")
    if g == "u04":
        ci, sh = gj.get("ci99_harf"), gj.get("harf_share_S11_S12")
        s = ("записей harf:* на V_k нет" if gj.get("diff_harf") is None else
             f"micro Δ (Б − А) по {gj['n_harf']} записям harf:* V_k: {gj['diff_harf']:+.2f} п. п., {100 * gj['level']:.0f} % ДИ "
             f"[{ci[0]:+.2f}; {ci[1]:+.2f}] (бутстреп записей в стратах V_k: исполнители фиксированы, записи внутри — с возвращением; "
             "фильтр механизма; решение о шаге — отбор на ГП по всем классам; перенос механизма букв на исполнителей не проверяют ни "
             f"отбор, ни подтверждение, разд. 4.3); из них у S11 + S12 (пилот) — {100 * sh:.1f} %: вывод о буквах держится "
             "в основном на двух исполнителях пилота")
        s += f"; Δ по всем классам на V_k (справочно, macro): {gj['diff_all_ref']:+.2f}" if gj.get("diff_all_ref") is not None else ""
        pl, cz = gj.get("palm_frames_lt_half_median"), gj.get("one_plus_cos_zero_visible")
        s += "" if pl is None else (f"; кадры с ладонью |9 − 0| < 0,5 медианы записи — {100 * pl:.2f} %, 1 + cos = 0 у видимых кистей — "
                                    f"{100 * cz:.3f} % (справочно)")
        return s + ("; ворота закрыты — **«U04 не проверена», а не «не прошла»**: бутстреп записей в стратах V_k при эффекте "
                    "≈ +5 п. п. на буквах открывается лишь в 0,16–0,58 (разд. 4.3)" if gj["decision"] == "skip" else "")
    return ""


def report(tags: list[str]) -> None:
    import m13_build
    off = m13_build.load("alluz")["offsets"]
    T = int(min(np.median(np.diff(off)), 384))
    import uzsohn_gates as G
    codes = json.loads(CODES.read_text())["signers"]
    rows, allms, pres, vk_of = [], [], {}, None
    for tag in tags:
        ms = [json.loads((RUNS / tag / f / f"seed{SEED}/metrics.json").read_text()) for f in FOLDS]
        vk_of = vk_of or {f: m["inner_val"] for f, m in zip(FOLDS, ms)}
        assert vk_of == {f: m["inner_val"] for f, m in zip(FOLDS, ms)}, tag
        sc = scores(tag)
        dec = OUT / f"decide_{tag}.json"
        dd = json.loads(dec.read_text()) if dec.is_file() else None
        tok = tuple(ms[0].get("input") or ())
        pre = RUNS / ms[0]["init"]["init_from"].split("runs/UzSohn/", 1)[1] if ms[0].get("init") else None
        pm = json.loads((pre.parent / "metrics.json").read_text()) if pre else None
        allms += ms
        if pm:   # свои предобучения этапа (48): контроль sb/all/seed48 и плечи sb_in_* (путь сверил scores()); полный бюджет — 13 800 шагов
            assert pm.get("seed") == SEED and full_budget(pm) and pm.get("steps") == PRE_STEPS, (
                pre, pm.get("seed"), pm.get("steps_run"), pm.get("steps"), "предобучение не то (затравка / бюджет / 13 800 шагов): "
                "в копию — его, прогоны отбора от него, decide_*.json этого шага и следующих и gate_*.json следующих шагов (решения "
                "жадные: следующий шаг сравнивается с текущей, ворота U01a и U04 считаются по ней; разд. 4.4)")
            pres[str(pre.parent)] = pm
        rows.append({"tag": tag, "variant": list(tok) or ["Sohn"], "macro_vk": round(float(np.mean(list(sc.values()))), 2),
                     "decision": dd, "params": ms[0]["params_keras_count"]["total"],
                     "params_stem": params_stem(ms[0]),
                     "ms_cpu": ms_per_record(ms[0]["flags"] if ms[0].get("flags") else {}, ms[0]["n_classes"], T, tokens=tok),
                     "gpu_h": round(sum(m["train_sec"] for m in ms) / 3600, 2),
                     "pre_gpu_h": round(pm["train_sec"] / 3600, 2) if pm else None, "pre": str(pre.parent.relative_to(ROOT)) if pre else None, "steps": int(sum(m["steps"] for m in ms)),
                     "examples_M": round(sum(m["steps"] for m in ms) * 512 / 1e6, 2), "gpu_mem_gb": max(m.get("gpu_max_mem_gb") or 0 for m in ms)})
    b = json.loads((OUT / "base.json").read_text())
    L = ["# UzSohn, этап 1 — отбор идей на внутренних V_k (короткий итог автору)", "",
         f"База: **{b['set']}** (H2 {b['H2_verdict']}: {b['H2']['diff']:+.2f} [{b['H2']['ci'][0]:+.2f}; {b['H2']['ci'][1]:+.2f}]). "
         f"Порог: Δ macro ≥ {THR:.3f} п. п. (max(1; δ = {DELTA:.3f})) и лучше у ≥ {MIN_BETTER}/10 V_k. Затравка {SEED}, "
         "1 прогон на фолд; **это отбор, не вывод** — числа V_k не главная гипотеза (подтверждение — 3 затравки на S_k, после решения автора).", "",
         FALSE_ACCEPT, "", "Записи S_k участвуют в отборе как V соседнего фолда (3 934 из 4 084): подтверждение H не независимо от "
         "отбора по исполнителям (разд. 2, угроза 11); ниже — решение каждого шага и ворот U04 без V_j = S_k (по 9 V_j ≠ S_k; у ворот "
         "U01a — только q, справка: ворота — только печать). S_k при "
         "этом остаётся в обучении 8 из 9 этих прогонов отбора и в прототипах и классах ворот их фолдов: «совпало» не значит, что "
         "отбор вложенный; полностью вложенный отбор (свой отбор на каждый k) не делается — выбор 0087.", "",
         f"| вариант (вход) | macro V_k | Δ к текущей | лучше | принята | параметры | из них stem | мс/запись ЦП (T = {T}) | ч ГП* отбор | ч ГП* предобуч. S+B | примеры, млн | ГП, ГБ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        d = r["decision"]
        L.append(f"| {r['tag']} ({'+'.join(map(str, r['variant']))}) | {r['macro_vk']} | "
                 f"{'' if not d else format(d['diff'], '+.3f')} | {'' if not d else d['better']} | "
                 f"{'база' if not d else ('да' if d['accepted'] else 'нет')} | {r['params']} | {r['params_stem']} | {r['ms_cpu']} | {r['gpu_h']} | {r['pre_gpu_h']} | {r['examples_M']} | {r['gpu_mem_gb']} |")
    fp = OUT / "facts_stage1.json"
    el = json.loads(fp.read_text()).get("elbows_by_signer", {}) if fp.is_file() else {}
    L.append("")
    for r in rows:
        d = r["decision"]
        if not d:
            continue
        a, c = scores(d["current"]), scores(d["candidate"])
        per = {f: c[f] - a[f] for f in FOLDS}
        L.append(nested_line(r["tag"], d["accepted"], per, vk_of, ok_delta, codes,
                             lambda s: float(np.mean(list(s.values()))), "Δ macro"))
        kk = {codes.get(vk_of[f]): per[f] for f in FOLDS}
        if d.get("idea") == "U04":    # harf:* на V_k — в основном у пилота: «лучше у k/10» перенос механизма букв не проверяет (4.3)
            rest = [v for s, v in kk.items() if s not in ("S11", "S12")]
            L.append(f"U04: Δ_k на V_k = S11 {kk.get('S11', float('nan')):+.2f}, S12 {kk.get('S12', float('nan')):+.2f} п. п., среднее по "
                     f"остальным {len(rest)} V_k {float(np.mean(rest)):+.2f} — записи harf:* на V_k в основном у S11, S12 (пилот), "
                     "поэтому «лучше у k/10» перенос механизма букв на исполнителей не проверяет (разд. 4.3)")
        if d.get("idea") == "U01a":   # угроза 7: у пилота S11, S12 локти часто ниже кадра (экстраполяция MediaPipe)
            low = "; ".join(f"{s} {100 * el[s]['frac_elbows_below']:.0f} %" for s in ("S11", "S12") if s in el)
            L.append(f"U01a: Δ_k на V_k = S11 {kk.get('S11', float('nan')):+.2f}, S12 {kk.get('S12', float('nan')):+.2f} п. п. — "
                     f"локти там часто ниже кадра (y > 1, доля значений: {low or '—'}; facts_stage1.json) и экстраполированы "
                     "MediaPipe почти без пометки видимостью (угроза 7)")
    sp = OUT / "stats_u02.json"
    if sp.is_file():
        L.append("\n" + u02_note(json.loads(sp.read_text())))
    for g in ("u01a", "u04"):
        gp = OUT / f"gate_{g}.json"
        if gp.is_file():
            gj = json.loads(gp.read_text())
            how = f"по правилу q ≥ {G.Q_MIN} (только печать, вариант (в) разд. 12 п. 5)" if g == "u01a" else "решение"
            L.append(f"\nЦП-ворота {g}: {how} {json.dumps(gj['decision'], ensure_ascii=False)}; правило: {gj['rule']}")
            L.append(gate_note(g, gj))   # у U01a — и q без V_j = S_k (справка): ворота ничего не решают
            pf = gj.get("per_fold")
            if g == "u04" and pf:
                L.append(nested_line("ворота U04", gj["decision"], pf, {f: v["vk"] for f, v in pf.items()},
                                     lambda s: G.u04_rule(s)[3], codes, lambda s: G.u04_rule(s)[0], "Δ (Б − А)"))
    g4 = OUT / "gate_u04.json"
    L += ["", u01b_note(json.loads(g4.read_text())["decision"] if g4.is_file() else None,
                        next((r["decision"]["accepted"] for r in rows if "u04" in r["variant"] and r["decision"]), None)),
          "", branch_note(allms + list(pres.values())), "", code_note(allms + list(pres.values()))]
    L += ["", "* часы ГП — время прогонов по часам (train_sec); где на ГП шло несколько прогонов, оно завышено: отбор — 2–3 прогона на "
               "одном ГП; предобучения контроля (sb/all/seed48) и U02-1 — по два (фаза 0; U02-1 иногда ещё рядом с отбором контроля); "
               "предобучения U02-2, U01a и U04 — по одному, не завышены. Столбец предобучения между строками поэтому не сравним; мера "
               "без искажения — примеры: у отбора шаги × 512 (столбец), у каждого предобучения 13 800 × 512 = 7,07 млн.",
          "При равном числе примеров у всех вариантов (тот же рецепт Sohn, шаги = 300 × ⌊реальных записей / 512⌋)."]
    for p in (ROOT / "reports/UZSOHN_stage1.md", OUT / "stage1.json"):
        if p.is_file():   # копии вместо перезаписи (правило автора): прежний итог — рядом, с датой
            p.rename(p.with_name(f"{p.stem}_{time.strftime('%Y%m%dT%H%M%S')}{p.suffix}"))
    (ROOT / "reports/UZSOHN_stage1.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (OUT / "stage1.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("\n".join(L))


def report2(cur: str, cand: str) -> None:
    """Этап 2 (UZSOHN_prereg_popravka2.md): U01b поверх Sohn против контроля этапа 1 → reports/UZSOHN_stage2.md и
    runs/UZSOHN/stage2.json (прежние — в копию с датой). Решение — decide_<cand>.json (правило этапа 1); числа — только V_k."""
    import m13_build
    T = int(min(np.median(np.diff(m13_build.load("alluz")["offsets"])), 384))
    codes = json.loads(CODES.read_text())["signers"]
    d = json.loads((OUT / f"decide_{cand}.json").read_text())
    assert (d["current"], d["candidate"]) == (cur, cand), d
    rows, ms_of, sc = [], {}, {}
    for tag in (cur, cand):
        ms = [json.loads((RUNS / tag / f / f"seed{SEED}/metrics.json").read_text()) for f in FOLDS]
        sc[tag] = scores(tag)
        pre = RUNS / ms[0]["init"]["init_from"].split("runs/UzSohn/", 1)[1]
        pm = json.loads((pre.parent / "metrics.json").read_text())
        assert pm.get("seed") == SEED and full_budget(pm) and pm.get("steps") == PRE_STEPS, (pre, "предобучение не то")
        ms_of[tag] = ms + [pm]
        tok = tuple(ms[0].get("input") or ())
        rows.append({"tag": tag, "variant": list(tok) or ["Sohn"], "macro_vk": round(float(np.mean(list(sc[tag].values()))), 2),
                     "params": ms[0]["params_keras_count"]["total"], "params_stem": params_stem(ms[0]),
                     "ms_cpu": ms_per_record({}, ms[0]["n_classes"], T, tokens=tok), "gpu_h": round(sum(m["train_sec"] for m in ms) / 3600, 2),
                     "pre_gpu_h": round(pm["train_sec"] / 3600, 2), "examples_M": round(sum(m["steps"] for m in ms) * 512 / 1e6, 2),
                     "gpu_mem_gb": max(m.get("gpu_max_mem_gb") or 0 for m in ms)})
    vk_of = {f: m["inner_val"] for f, m in zip(FOLDS, ms_of[cand])}
    assert vk_of == {f: m["inner_val"] for f, m in zip(FOLDS, ms_of[cur])}, "V_k плеча ≠ V_k контроля"
    per = {f: sc[cand][f] - sc[cur][f] for f in FOLDS}
    ok = d["accepted"]
    L = ["# UzSohn, этап 2 — U01b на внутренних V_k (короткий итог автору)", "",
         "U01b: вход Sohn (118 точек, бит в бит) + 84 столбца в конце того же stem, сырыми: на кисть 21 точка (x, y · H/W), (точка − "
         "запястье) / ладонь, ладонь — медиана |9 − 0| по кадрам входа с точками 0 и 9 (после аугментации и масок, первые 384 кадра); "
         "без поворота и скоростей; NaN → 0, обрезка [−10; 10]; кисть без таких кадров — её 42 столбца 0; 4 столбца «запястье − "
         "запястье» тождественно 0 (оставлены). Определение — 0153, к1 0136 п. 5, к3 0039 разд. 5; ворот ЦП нет.", "",
         f"Правило этапа 1 (разд. 3): Δ macro ≥ {THR:.3f} п. п. и лучше у ≥ {MIN_BETTER}/10 V_k против контроля этапа 1 ({cur}; путь "
         "Sohn — бит в бит с базой, тест test_stage1_default_path_bit_exact_vs_base). Затравка 48, 1 прогон на фолд; **это отбор, не "
         "вывод**.", "", RISK2, "",
         f"| вариант (вход) | macro V_k | Δ к контролю | лучше | принята | параметры | из них stem | мс/запись ЦП (T = {T}) | ч ГП* отбор | ч ГП* предобуч. S+B | примеры, млн | ГП, ГБ |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r, dd in zip(rows, (None, d)):
        L.append(f"| {r['tag']} ({'+'.join(r['variant'])}) | {r['macro_vk']} | {'' if not dd else format(dd['diff'], '+.3f')} | "
                 f"{'' if not dd else dd['better']} | {'контроль' if not dd else ('да' if ok else 'нет')} | {r['params']} | "
                 f"{r['params_stem']} | {r['ms_cpu']} | {r['gpu_h']} | {r['pre_gpu_h']} | {r['examples_M']} | {r['gpu_mem_gb']} |")
    kk = {codes.get(vk_of[f], "?"): per[f] for f in FOLDS}
    rest = [v for s, v in kk.items() if s not in ("S11", "S12")]
    L += ["", "Δ_k по V_k, п. п.: " + ", ".join(f"{s} {v:+.2f}" for s, v in sorted(kk.items())),
          f"U01b: Δ_k на V_k = S11 {kk.get('S11', float('nan')):+.2f}, S12 {kk.get('S12', float('nan')):+.2f} п. п., среднее по "
          f"остальным {len(rest)} V_k {float(np.mean(rest)):+.2f} — признаки формы кисти, а записи harf:* на V_k в основном у S11, S12 "
          "(как у U04, разд. 4.3)",
          nested_line(cand, ok, per, vk_of, ok_delta, codes, lambda s: float(np.mean(list(s.values()))), "Δ macro"), "",
          ("**U01b ПРИНЯТА** (серия UzSohn на ALL-UZ: UzSohn = Sohn + U01b); подтверждение (3 затравки на S_k, ≈ 23–25 ч ГП) — только "
           "по решению автора." if ok else "**U01b не принята** → UzSohn = Sohn (этапы 1–2 не приняли ничего)."),
          "Турнир (толкование к2 решения автора 16.11, вариант (а′)): U01b — в списке кандидатов Э2 при любом итоге здесь; отбор — только "
          "на валидации Slovo + Bukva (П5), корпус — только тест.", "",
          "Код U01b: " + code_note(ms_of[cand]), "Код контроля (этап 1): " + code_note(ms_of[cur]), code_vs(ms_of[cur], ms_of[cand]), "",
          "* часы ГП — по часам (train_sec); отбор U01b — по 2 прогона на ГП, контроль — по 3 (этап 1), поэтому часы между строками не "
          "сравнимы; мера без искажения — примеры (шаги × 512)."]
    for p in (ROOT / "reports/UZSOHN_stage2.md", OUT / "stage2.json"):
        if p.is_file():   # копии вместо перезаписи (правило автора)
            p.rename(p.with_name(f"{p.stem}_{time.strftime('%Y%m%dT%H%M%S')}{p.suffix}"))
    (ROOT / "reports/UZSOHN_stage2.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    dj = {**d, "per_vk": {codes.get(vk_of[f], "?"): v for f, v in d["per_vk"].items()}}   # S-коды V_k (decide_*.json — названия, только ПК)
    (OUT / "stage2.json").write_text(json.dumps({"rows": rows, "decision": dj}, ensure_ascii=False, indent=1))
    print("\n".join(L))


# ---------------------------------------------------------------- Э2 турнира (поправка 4 к reports/TURNIR_prereg.md, 22.3–22.10)
# 22.3: шаг → его рычаг (вход (а′) или № 2–4: r2 — токены частей, r3 — EMA, r4 — длинный хвост); жадно, в этом порядке, поверх текущей
E2_STEPS = {"U02-1": "u02s", "U02-2": "u02c", "U01a": "u01a", "U04": "u04", "U01b": "u01b", "r2": "r2", "r3": "r3", "r4": "r4"}
E2_CELLS = [f"{s}/fold{k}" for s in ("slovo", "bukva") for k in range(1, 6)]   # 22.4: ячейка — (набор, фолд)


def levers(m: dict) -> set[str]:
    """Рычаги прогона по metrics.json m13_train: токены входа и № 2–4 (22.11)."""
    return set(m["input"]) | {r for r, on in (("r2", m["flags"]["part_tokens"]), ("r3", m["ema"]), ("r4", m["balanced"])) if on}


def e2_runs(suf: str) -> dict[str, tuple[dict, list]]:
    """Ячейка → (metrics.json, [(sample_id, signer_id, верно)] из predictions_val.csv) прогонов runs/UzSohn/uzsohn_e2_<набор>_<suf>/
    <фолд>/seed48. Счёт не на валидации или файл предсказаний без _val — «!!!», стоп: такой каталог стенд прочёл бы как тест (22.8)."""
    out = {}
    for c in E2_CELLS:
        s, f = c.split("/")
        d = RUNS / f"uzsohn_e2_{s}_{suf}" / f / f"seed{SEED}"
        m = json.loads((d / "metrics.json").read_text())
        bad = sorted(p.name for p in d.iterdir() if p.name.startswith(("predictions", "probs", "logits")) and not p.stem.endswith("_val"))
        if m.get("score_on") != "val" or bad:
            sys.exit(f"!!! {d}: score_on = {m.get('score_on')!r}, файлы без _val: {bad or '—'} — не прогон --val-only (22.8), стоп")
        # 22.2, 22.5: с нуля (без --init), затравка 48, полный бюджет рецепта; из идей 0107 — только № 2 (9)
        assert (m["set"], m["fold"], m["seed"], m["init"]) == (s, f, SEED, None) and full_budget(m) and set(m["variant"]) <= {9}, d
        with open(d / "predictions_val.csv", newline="", encoding="utf-8") as fh:
            r = [(x["sample_id"], x["signer_id"], int(x["correct"])) for x in csv.DictReader(fh)]
        assert r and sum(x[2] for x in r) / len(r) == m["top1"], (d, "predictions_val.csv ≠ top1 (главная точка best.pt)")
        out[c] = (m, r)
    return out


def e2_cells(R: dict, drop=(), macro: bool = False) -> dict[str, float]:
    """Счёт ячеек, % (22.4): micro top-1 по записям валидации — решение; macro — среднее точностей исполнителей (signer_id) —
    только печать; drop — коды исполнителей вне счёта (печать «без группы j», 22.7; ячейка без записей выпадает)."""
    out = {}
    for c, (_, r) in R.items():
        by = {}
        for _, s, ok in r:
            if s not in drop:
                by.setdefault(s, []).append(ok)
        if by:
            out[c] = 100 * float(np.mean([np.mean(v) for v in by.values()] if macro else [ok for v in by.values() for ok in v]))
    return out


def e2_row(a: dict, b: dict) -> dict:
    """Правило 22.4 (judge: Δ̄ ≥ THR и Δ > 0 у ≥ 6 ячеек; условие 3 снято) на счёте ячеек текущей a и кандидата b → печать."""
    d, diff, better, ok = judge(a, b)
    mean = lambda x: round(float(np.mean(list(x.values()))), 3)
    return {"current": mean(a), "candidate": mean(b), "diff": round(diff, 3), "better": f"{better}/{len(d)}", "accepted": ok,
            "per_cell": {c: round(x, 2) for c, x in d.items()}}


def m13_spread() -> dict:
    """22.4, справочно (правило не меняет): val top-1 (micro) на лучшей эпохе (min val_loss, как best.pt) из log.csv M13
    runs/UzSohn/<набор>/<фолд>/seed42–46, среднее 5 фолдов; пометка, если наибольшая разница двух затравок из 42–44 по среднему
    наборов больше порога. Только log.csv: metrics.json M13 несёт тест (22.1, второй путь теста)."""
    def best(p: Path) -> float:   # первая эпоха минимума — как ModelCheckpoint (vloss < best)
        with open(p, newline="") as fh:
            return float(min((x for x in csv.DictReader(fh) if x["val_loss"]), key=lambda x: float(x["val_loss"]))["val_top1"])
    logs = lambda s, k: [RUNS / s / f"fold{i}" / f"seed{k}/log.csv" for i in range(1, 6)]
    v = {s: {k: 100 * float(np.mean([best(p) for p in logs(s, k)])) for k in range(42, 47) if all(p.is_file() for p in logs(s, k))}
         for s in ("slovo", "bukva")}
    m = [(v["slovo"][k] + v["bukva"][k]) / 2 for k in (42, 43, 44) if k in v["slovo"] and k in v["bukva"]]
    gap = max(m) - min(m) if len(m) == 3 else None
    return {"val_top1": {s: {k: round(x, 3) for k, x in z.items()} for s, z in v.items()},
            "range": {s: round(max(z.values()) - min(z.values()), 3) if z else None for s, z in v.items()},
            "gap_42_44": None if gap is None else round(gap, 3),
            "note": "порог ниже шума затравок M13" if gap is not None and gap > THR else None}


def e2(cur: str, cand: str, step: str) -> None:
    """Шаг отбора Э2 (22.3–22.5): CAND = текущая CUR + рычаг шага, 10 ячеек, затравка 48, свой контроль; решение — micro (22.4),
    рядом — macro, счёт last.pt, «без группы j» (22.7), разброс M13, доля ветки Sohn у u02*, параметры входа (tok у № 2)."""
    p = OUT / f"decide_e2_{step}.json"
    if p.is_file():
        sys.exit(f"!!! {p} уже есть — решение шага не перезаписывается (22.10 п. 2)")
    steps = list(E2_STEPS)
    i = steps.index(step)
    q = json.loads((OUT / f"decide_e2_{steps[i - 1]}.json").read_text()) if i else {"accepted": False, "current": "v0"}
    assert cur == q["candidate" if q["accepted"] else "current"], f"22.3: шаг {step} — поверх итога прошлого шага, а не {cur}"
    A, B = e2_runs(cur), e2_runs(cand)
    (lc,), (ln,) = ({frozenset(levers(m)) for m, _ in R.values()} for R in (A, B))   # у 10 ячеек тега — один набор рычагов
    assert E2_STEPS[step] not in lc and ln == lc | {E2_STEPS[step]}, f"22.3: {cand} {sorted(ln)} ≠ {cur} {sorted(lc)} + {step}"
    assert all(sorted(x[0] for x in A[c][1]) == sorted(x[0] for x in B[c][1]) for c in E2_CELLS), "записи валидации ≠ у текущей"
    mic = e2_row(e2_cells(A), e2_cells(B))
    ok = mic["accepted"]
    groups = json.loads((ROOT / "splits/slovo_p5.json").read_text())["groups"]   # исполнитель → группа теста Slovo (фолд = группа + 1)
    wg = {}
    for j in range(1, 6):   # 22.7: коды группы теста Slovo j — вне валидации Slovo и Bukva во всех фолдах
        drop = {s for s, g in groups.items() if g == j - 1}
        wg[j] = e2_row(e2_cells(A, drop), e2_cells(B, drop))
    bad = [j for j, x in wg.items() if x["accepted"] != ok]
    last = lambda R: {c: 100 * m["top1_last"] for c, (m, _) in R.items()}
    res = {"step": step, "current": cur, "candidate": cand, "levers_current": sorted(lc), "levers_candidate": sorted(ln),
           "accepted": ok, "threshold": round(THR, 4), "delta": DELTA, "micro": mic,
           # печать рядом, на решение не влияет (22.4, 22.7)
           "macro_ref": e2_row(e2_cells(A, macro=True), e2_cells(B, macro=True)), "last_ref": e2_row(last(A), last(B)),
           "without_group": wg,
           "without_group_line": f"без группы j (22.7): решение совпало у {5 - len(bad)}/5 групп теста Slovo" + (
               f"; не совпало у групп {', '.join(map(str, bad))} — у пар UzSohn − X на Slovo пометка «отбор не вложенный»" if bad else ""),
           "m13_seed_spread": m13_spread(), "u02_sohn_branch": branch_note([m for m, _ in B.values()]),
           # код 20 прогонов шага: цепочка сверяет его с хешем один раз, до пробы; правка за ≈ 32–36 ч Э2 — «ВНИМАНИЕ» (как этап 1)
           "code": code_note([m for R in (A, B) for m, _ in R.values()]),
           "params": {t: {"total": {s: R[f"{s}/fold1"][0]["params_keras_count"]["total"] for s in ("slovo", "bukva")},
                          "params_stem": params_stem(R["slovo/fold1"][0])} for t, R in ((cur, A), (cand, B))}}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(p, "x", encoding="utf-8") as fh:   # «x»: и между проверкой выше и записью файл не перезаписывается
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print(json.dumps({"step": step, "accepted": ok, "diff": res["micro"]["diff"], "better": res["micro"]["better"],
                      "diff_macro_ref": res["macro_ref"]["diff"], "without_group": f"{5 - len(bad)}/5",
                      "m13": res["m13_seed_spread"]["note"], "code_ok": not res["code"].startswith("**")}, ensure_ascii=False))
    sys.exit(0 if ok else 10)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "base":
        base()
    elif c == "ideas":
        ideas(sys.argv[2])
    elif c == "decide":
        decide(*sys.argv[2:5])
    elif c == "selftest":
        selftest()
    elif c == "report":
        report(sys.argv[2:])
    elif c == "report2":
        report2(*sys.argv[2:4])
    elif c == "e2":
        e2(*sys.argv[2:5])
    else:
        sys.exit(f"неизвестная команда {c}")
