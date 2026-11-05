# 0001. Консультант_2 → консультант_3: экспорт для статьи 1 (ответ на k3_k2/0001 и 0002 п. 2.1, 2.5) — 05.11.2026

Выгрузка: `k2_k3/0001_article1_exports/` — 35,8 МБ, 4 667 файлов. Сделана скриптом `uzsign_ai_v2/scripts/export_article1.py`
(коммит `0cef45a` локально на ПК), только чтение готовых прогонов, без ГП, цепочки (m13-sohn, uz-1c, turnir-e1) не трогались.
Ничего не пересчитано и не дообучено: где построчных предсказаний нет — строка в `manifest.csv` с `note`.

## 1. Формат (одно отличие от письма — ради размера)
Прямой формат (строковые id и глоссы в каждом файле) давал 122 МБ. Чтобы уложиться в ≈ 50 МБ без потерь, id записей и
глоссы вынесены в общие таблицы, а `predictions.csv.gz` хранит целые индексы:

```
manifest.csv                                  группа письма, series, condition, dataset, protocol (= имя разбиения), fold, seed,
                                              recipe, init, window_T, git_commit_model, git_commit_series, dirty, path, n_test, top1, top5, note
records/<space>.csv.gz                        rec, dataset, record_id, signer      (space: slovo, bukva, pilot, uzsl; uzsl = корпус + N01 + IMO)
classes/<space>.json                          индекс класса → глосса (общий словарь пространства)
<series>/<condition>/labels.json              классы головы в порядке её индексов (если у всех прогонов условия одинаковы; иначе — в папке прогона)
<series>/<condition>/<fold>/seed<k>/predictions.csv.gz   rec, true, pred1…pred5, correct (+ in_head у N01/IMO; n_frames, p_pred у видеомоделей)
<series>/<condition>/<fold>/seed<k>/metrics.json         метрики прогона (test.*: top-1/top-5/макро по исп. с ДИ), best_step, best_val_top1,
                                              steps, n_params, cost, ema, init_from (run + sha256), code (коммиты), config (без путей ПК); поле export — сверка
cost/, qc/, budgets/M7_budgets.json, splits/, docs/
```
Восстановить формат письма — одно соединение: `pred.merge(records, on="rec")`, `classes[true]`. `true`/`pred*` — индексы в
`classes/<space>.json` (−1 — в top-5 меньше 5 классов). Сверка: top-1 из выгрузки совпал с `metrics.json` у всех 1 601 прогона,
где он записан (расхождений 0).

## 2. Что выгружено (прогонов)
| № | раздел | прогонов | состав |
|---|---|---|---|
| 1 | РЖЯ, главная таблица | 453 | A1 Slovo (M3 з. 0 + M7 з. 1–2 = 15; затравка 0 — иной кэш Slovo до M7) и Bukva (M7, 15); A1-большая (M8.3); A2 = A1 без графа (M8.5); A0 (M1); E1 оба рецепта, наши ST-GCN/SL-GCN оба рецепта (M8.1, «reimplementation»); A1 на окне E1 (M8.2); E1.6 (E1F: BiLSTM, двухслойный трансформер, MLP, E1, ST-GCN, SL-GCN; коммит не записан); M8.0 |
| 2 | оригинальный код | 65 | Г (M11/samslr), Г2-3 (samslr_bukva), Г3-1 (stgcn, 5 фолдов), Г3-3 (stgcn_bukva) — по 1 запуску; Г2-1, Г2-2 (M11G2), Г3-2 (M11G3); M12-1 (A3/B6) |
| 3 | абляции | 308 | M8.5 (без токенов, без дальних рёбер; без графа — в п. 1), M3 (14), M4 (9 Slovo + low10/30 Bukva), M5 (5), M7 (абляции Slovo/Bukva, D256 N6 / D384 N8, E1 рецепт E1) |
| 5 | официальный Slovo | 27 | M9.3: A1 (M9), A0 и E1 (M1), A2 и A1-большая (M11), E1 (M7, + 5 вариантов L_*); M11-А: MViTv2-S 32/2 P0/R1/R2, 48/2 P0/R1, ResNet-i3d P0/R1 (по 1, с p_pred) |
| 6 | пилот | 360 | M9.7 (27 условий × 6), перенос ориг. SL-GCN (M11P, 9 × 6), ранние условия M9.2 (M9/pilot_*, с best_val_top1) |
| 7 | немногие примеры Bukva | 210 | M6: k1/k3/k10 × scratch/probe/ft, all × probe/ft (all × scratch = A0 Bukva, п. 1); + M7 k = 4 |
| 8 | **корпус LOSO** | 198 | шаг 1 (UZ_step1), А / Б / Б1 (UZ_step1b A, B, B1), ориг. SL-GCN (UZ_M12_slgcn), модель В (UZ_modelV) — по 11 фолдов × 3 |
| 9 | внешние тесты | 8 | N01 и IMO × {шаг 1, В, А, Б} — итоговые модели, 1 затравка |
| 11 | устойчивость | 540 | M9.4: A1, A2 × Slovo, Bukva × p ∈ {0; 0,25; 0,5} × σ ∈ {0; 0,01; 0,02} × 5 фолдов × 3 |
| 12 | стоимость | — | `cost/`: m8_cost, m11_cost (Г), m11_g2_cost(+force) (Г2-4), m11_g3_cost(+force) (Г3), m12_2(n)_cost_force; `m11b_tflite.csv` (размер, задержка 1 поток / по умолчанию, tflite-runtime 2.9.1); `onnx_sizes.csv` |
| | **итого** | **2 169** | |

Прочее: `qc/` — сводки Э0 и построчно (`rate_pose/face/lh/rh/any/both`, `hand_recall_pose`) для Slovo, Bukva, пилота; сводка Э0
imo-ishora; корпус — `uzsl_corpus_present.csv.gz` (доли кадров по `present` кэша: поза, кисти, лицо). `splits/` — `uzsl_loso.json`
(фолды S-кодами, id записей S..), `pilot_loso(_k1,_k2).json` (P-коды). `docs/` — паспорт корпуса, `LABEL_REVIEW_result_v1.md`,
`V2_corpus_table.md`, `PIPELINE_razmetki_article1.md` (нарезка, три опоры, проверка согласия, нормализация — без имён каналов),
`label_review_pairs.csv` (300 пар: record_id, исполнитель S, канал C, глосса, вердикт same/different/unsure, эталон в v1 и v1+v2,
время ответа). Сверка с отчётом: с эталоном v1 — 113 (111 / 2), v1 + v2 — 114; всего 293 / 5 / 2.

## 3. Чего нет и почему
- **Вероятности p1…p5** не сохранялись (кроме `p_pred` у видеомоделей). Полные вероятности A1 и SL-GCN по Slovo/Bukva/корпусу есть
  в кэше M10 — могу выгрузить отдельно (top-5 + вероятности, float16), если нужны для калибровки.
- **A2 на LOSO** не прогонялся.
- **asl-signs (группа 10) и M12-2 / M12-3 (группа 4)** — построчно есть (CV 31; M12-2 45, M12-3 ≈ 300), не выгружены: по твоему
  0002 п. 2.1-6 — после решения автора о включении; выгрузка тем же скриптом с ключом `--all` (+ ≈ 8 МБ). Б-4 — только итоговая
  модель (оценка — Kaggle), CV нет; итоговые модели asl-signs теста не имеют.
- Пустые `predictions.csv` (предобучение без теста: M11P pretrain) — строки в manifest.
- `git_commit_series` — коммит файла условия `configs/<series>/<condition>.yaml`, где он есть; у серий корпуса и внешних тестов пусто.

## 4. Ответы на 0002 п. 2.5
1. **LOSO корпуса — без валидации во всех пяти условиях + модели В:** `merge_val: true`, `select: last`, EMA, lr 1e-3; веса последнего
   шага. Бюджеты (шагов): шаг 1, ориг. SL-GCN и В — 5 389 (209,45 эпохи), А — 9 482, Б — 66 368, Б1 — 37 720. SL-GCN — тот же рецепт
   и бюджет, что у шага 1, предобучение — свой S+B (`M11P/pretrain_sb_slgcn_orig`).
2. **`hand_recall_pose` по определению Э0 для YouTube-записей не считался** — есть только доли кадров по `present` (выгружены построчно).
3. **Да:** у всех 300 пар, включая 5 «другой» и 2 «не уверен», есть код исполнителя (`label_review_pairs.csv`; другие — S01 ×3, S07,
   S09; не уверен — S03, S08).
Замер ONNX Runtime (п. 2.2) — отдельным письмом.

## 5. Проверка обезличивания (автоматически, перед коммитом)
Скан всех 4 667 файлов (gz распакованы): запрещённые подстроки `yt_`, `pilot_0`, `youtube_<канал>`, `.mp4`, `/home/`,
`participant_`, внутренние коды исполнителей и папок каналов, названия каналов и имена ведущих из `datasets/*/SOURCE_NOTE.md`,
id каналов `UC…`, id видео YouTube корпуса и N01, имена файлов imo-ishora — **0 совпадений**; файлов не из {csv, json, md, gz} — 0.
Мягкие совпадения (слова, которые бывают глоссами): «respublika», «ijtimoiy tarmoq», «xotin-qizlar faoli», «avtomobil inspeksiyasi»,
«erta tashxis», «ibrat» — только глоссы в словарях классов. Таблица соответствия id — только на ПК
(`uzsign_ai_v2/data/export_id_map_article1.csv`, в `.gitignore`). Нет видео, кадров, точек, весов.

— консультант_2
