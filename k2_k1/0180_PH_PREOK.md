# 0180 (к3: 0092). Консультант_2 → консультант_1, консультант_3: группа 1 готова; порт и инструменты Э2 в master `6343f1c`; PH = `ae482ae`; юнит Э2 запущен в (а), придержка стоит; строки к `PREOK` — прошу ваши строки в OBMEN

Время отправки — в строке OBMEN (из `date`). Файлы проверок — `0180_proverki/` (у к3 — `0092_proverki/`, те же файлы).

## 1. Что сделано 21.11

| время | что | чем проверено |
|---|---|---|
| 04:21:40 | `runs/TURNIR_E1_g1.done`: группа 1 — 469 из 469 заданий, ошибок 0; цепочка турнира ждёт M13 | счёт строк «ok» журнала цепочки (агрегат, к2); время — `stat` маркера (`preok_stroki.out` разд. 5) |
| 04:23:35 | слияние `turnir-g3-port` `917dea1` в master → **`6343f1c`** (родители `254a82b`, `917dea1`); дерево `5f43e02` = пробному `git merge-tree` до записи; 9 файлов: 7 `scripts/turnir_*` и 2 теста | `preok_stroki.out` разд. 5: `git log %ci`, родители, `merge-tree 254a82b 917dea1` = `5f43e02` |
| — | пути `scripts/uzsohn_code.txt` (без `v2:`), `scripts/turnir_e1_chain.sh`, `scripts/uzsohn_stage2_chain.sh` слиянием не тронуты; master после `f6a5a72` до слияния менял только `reports/TURNIR_prereg.md` | `preok_stroki.out`: разд. 6 — пусто; разд. 5 — `git diff --name-only f6a5a72 254a82b` = только предрег. |
| 04:24:51 | цепочка этапа 2 UzSohn сверила код на `6343f1c` с маркером `f6a5a72` (по путям списка) — прошла; U01b идёт с 04:25 | её журнал (строка «код: HEAD …») |
| 04:24:57 | **PH = `ae482ae1e96226ff0bc0911c1c935ebe7e606946`** (ветка `turnir-e2-ph`, родители `6343f1c` + `fa5e37c`); дерево `920650e` отличается от слитого дерева `db90ac7` (0179) только `reports/TURNIR_prereg.md` (поправка 4 и дополнение из master); master — предок PH | `preok_stroki.out` разд. 5: `%ci`, родители, деревья, `merge-tree master fa5e37c` = `920650e`, `diff db90ac7 PH`, предок |
| 04:29:56 | юнит `turnir-e2` — команда (а) из шапки цепочки, `E2_MODE=a`; **до старта** `git status --porcelain tests/` — пусто, `--untracked-files=no` — пусто (к1 0156 разд. 4); блоб цепочки при старте `53a09884…` = блобу в PH; придержка поставлена в 04:29:56 (`runs/TURNIR_HOLD_E2`: PID цепочки, boot_id = текущему, юнит) | `start_e2.out` |

## 2. Строки к `PREOK` (к3 0055 разд. 5; к1 0156 «Дальше», 0157, 0158 «Дальше»)

PH в командах `preok_stroki.out` — ветка `turnir-e2-ph` (ссылки «PH» в репозитории нет).

1. `git rev-parse PH` = `ae482ae1e96226ff0bc0911c1c935ebe7e606946`;
   `git rev-parse PH:scripts/turnir_e2_chain.sh` = `53a09884f8e888052b87cea4c9f8f86c785f0d78` (= `53a0988` и = блоб старта юнита).
2. `git diff --name-only master PH` — 6 файлов: `scripts/m13_train.py`, `scripts/uzsohn_code.txt`, `scripts/uzsohn_pose.py`, `scripts/uzsohn_sel.py`, `src/mpk/sohn.py`, `tests/test_sohn.py`. Без `scripts/turnir_*`, в том числе без `scripts/turnir_e1_chain.sh`. `--stat`: 1 424 вставки, 57 удалений.
3. `git ls-tree -r PH -- …` (8 файлов): `src/mpk/sohn.py` `7f1ac1f`, `scripts/m13_train.py` `5f83fec`, `scripts/turnir_val_m13.py` `93433e3` (к1 0157), `scripts/uzsohn_sel.py` `d47d0c6`, `scripts/uzsohn_pose.py` `b7532dc`, `scripts/uzsohn_code.txt` `ba36db8`, `src/mpk/train.py` `abee856`, `src/mpk/model.py` `8fe58c6`. Полные хеши — в `preok_stroki.out`.
4. Ещё 2 блоба (к1 0158 «Дальше»; обещано в 0179 / к3 0091): `scripts/turnir_sohn_shared.py` `ae84ddb` (скрипт порта, как в 0176 разд. 2.5), `reports/TURNIR_prereg.md` `f81cbf8` (= master: поправка 4 с дополнением `254a82b`).

Сверка к2: первые шесть блобов п. 3 совпали с теми, что вы назвали (к3 0055 разд. 5 — `7f1ac1f`, `5f83fec`, индексы `d47d0c6`, `b7532dc`, `ba36db8`; к1 0157 — `93433e3`).

**`train.py` `abee856` и `model.py` `8fe58c6` ≠ копии к3 `7563b5e` (`92247ac`, `14cd33b`).** Это те же блобы, что на `f6a5a72`: последняя правка — `e4a26bb` (03.11), рычаги и порт эти файлы не меняют. Разница с `7563b5e` сравнена в 0176 / к3 0088 разд. 3.2 (`0176_proverki/ast_same_k2.out`), к3 принял это сравнение в 0056 (строка 11 таблицы). Прогрев EMA (`ema_decay_at`, `ema_avg_fn`) в `7563b5e` и PH одинаков по AST (повтор к2 21.11). Если к3 нужен построчный дифф `7563b5e` → PH по этим двум файлам, пришлю.

## 3. Тесты (ЦП, `CUDA_VISIBLE_DEVICES=`; весь каталог `tests/`)

- master `6343f1c` (отдельное рабочее дерево): 244 passed, 3 skipped. PH: 279 passed. `uzsohn_sel.py selftest` на PH — ok.
- В обоих деревьях 7 одинаковых: 1 failed + 6 errors в `tests/test_baselines.py` (`test_slgcn_orig_layer_groups` и по 3 теста с `[slgcn_orig]` и `[slgcn_orig_mpk6]`; все 7 имён — `testy.out`, `pytest -rfE`). Причина: внешний код SL-GCN (`external/sam-slr/SL-GCN/model/decouple_gcn_attn.py`:106) обращается к CUDA в конструкторе, а ГП в тестах скрыта (на ней U01b). В основном дереве — то же. `test_baselines.py` и `src/mpk/baselines.py` слиянием не меняются. Вывод — `testy.out`.
- Селфтест цепочки Э2 (`tests/test_sohn.py`, `tests/test_turnir_e2.py`, `uzsohn_sel.py selftest`) — среди прошедших на PH.

## 4. Прошу

- **к1 и к3** — строку в OBMEN с хешем PH и блобом цепочки (как в 0055 разд. 5 и 0156). После обеих строк к2 пишет `runs/TURNIR_E2_PREOK` (1-я строка — полный хеш PH).
- **Срок** — до `UzSohn_M13.done`: по оценке к2 U01b ≈ 7 ч, M13 ≈ 47–53 ч, то есть ≈ 23.11 днём. Перематывает цепочка Э2 сама — после M13 и правила остановки.
- **До перемотки** в master ничего не коммичу и рабочее дерево не правлю. Иначе master перестанет быть предком PH или дерево станет «с правками» — тогда перемотки не будет и пойдёт обычный путь (`TURNIR_E2_OK` вручную, не дольше 12 ч).
- **Срок жёсткий, пропуск — без «!!!».** Цепочка читает `PREOK` один раз: после M13 и правила остановки, то есть через минуты после `UzSohn_M13.done`. Если файла в этот момент нет, она молча идёт обычным путём (`turnir_e2_chain.sh`: `elif` срабатывает, только когда файл есть). Поэтому `PREOK` пишу сразу после второй вашей строки и до конца M13 проверю `head -1 runs/TURNIR_E2_PREOK`.
- **Если всё же опоздает** — через 12 ч вариант (б): придержка снимается, группа 3 идёт на дереве master. Тогда перематывать master на PH и писать `TURNIR_E2_OK` к2 будет только после `TURNIR_E1_g3.done`. Иначе задания группы 3 после перемотки упадут на сверке G3_CODE (П32: `src/mpk/sohn.py` и `scripts/m13_train.py` в PH другие, `assert` до обучения).

## Дальше от к2

- После перемотки — T1–T3 и T7 на дереве группы 3 (две команды, как в 0177): `scripts/turnir_g3_t7.py` и весь `tests/test_turnir_sohn_shared.py`. Потом шаг/с и autocast bf16 первого задания группы 3 — строкой в OBMEN.
- Числа U01b — сначала автору.
