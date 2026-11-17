#!/usr/bin/env bash
# Турнир, этап Э1 (reports/TURNIR_prereg.md, поправки 1–3): группа 1 → [ГП отдаётся M13: m13-sohn-2 / m13_chain_v3.sh,
# ждём runs/UzSohn_M13.done] → правило остановки UzSohn (ЦП) → группа 3 → честная M10-1b (уже сделана — пропуск) → группа 2
# → вероятности, сверка, сводка (ЦП). Внутри групп — «корпус, перенос» первыми (turnir_train.py).
# Старт — когда (1) есть runs/TURNIR_E1_GO (возвращает юнит turnir-go-after-1c после шага 1c), (2) M13 не занимает ГП:
# m13-sohn неактивен и есть runs/UzSohn_E0_FAIL (Э0 не пройден — продолжать с пометкой, 0078) или UzSohn_M13.done,
# (3) python на ГП нет 10 мин подряд, (4) этап 1 UzSohn не идёт и не стоит на ошибке: нет runs/UZSOHN_PREREG_OK или есть
# runs/UZSOHN_stage1.done (uzsohn_chain.sh маркер турнира не держит).
# Выходные (0064/0065): перед КАЖДЫМ новым прогоном — scripts/weekend_gate.sh (раннеры вызывают его сами); идущие прогоны
# не обрываются. Готовое пропускается (metrics.json / npz); маркеры — только при успехе шага. Повтор юнита продолжает с места.
#   cd ~/uzsign/osnovnaya_model && systemd-run --user --unit=turnir-e1 --collect -p WorkingDirectory=$PWD -p MemoryMax=40G \
#     -p MemorySwapMax=0 -E PATH="$PATH" -E HOME="$HOME" bash -c "bash scripts/turnir_e1_chain.sh >> runs/turnir_e1_chain.log 2>&1"
export PATH=$HOME/.local/share/mise/installs/python/3.14.7/bin:$PATH  # урок 30: каталог 3.14.7, не ссылка 3.14 (её двигает omarchy update → mise up); голый python с 11.11 — системный без numpy/torch
python -c 'import numpy, torch' 2> /dev/null || { echo "!!! ОШИБКА: $(command -v python) без numpy/torch (урок 30, ERRORS_LESSONS.md) — стоп"; exit 1; }
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../uzsign_e1/src
say() { echo "[$(date '+%F %T')] $*"; }
gpu_busy() { nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -q python; }  # только python: браузер/рабочий стол тоже бывают в списке (урок 27)
m13_done() { [ -f runs/M13.done ] || [ -f runs/UzSohn_M13.done ]; }
m13_paused() { m13_done || [ -f runs/UzSohn_E0_FAIL ]; }  # M13 остановилась на воротах Э0 и ждёт группу 1 (0078)
ready() {
  [ -f runs/TURNIR_E1_GO ] || return 1
  m13_paused || return 1
  systemctl --user is-active -q m13-sohn && return 1
  # UzSohn этап 1 маркер турнира не держит (uzsohn_chain.sh; UZSOHN_prereg разд. 14): разрешён и не завершён — ждать (урок 28)
  { [ -f runs/UZSOHN_PREREG_OK ] && [ ! -f runs/UZSOHN_stage1.done ]; } && return 1
  ! gpu_busy
}
J=${TURNIR_JOBS:-2}; JS=${TURNIR_JOBS_SMALL:-3}

say "=== Э1: жду runs/TURNIR_E1_GO, M13 на паузе (UzSohn_E0_FAIL) или готова, неактивности m13-sohn и свободной ГП 10 мин"
while true; do
  while ! ready; do sleep 60; done
  ok=1; for i in $(seq 10); do sleep 60; ready || { ok=0; break; }; done
  [ $ok = 1 ] && break
done
say "старт; план:"; python scripts/turnir_train.py plan --groups 1 3 2

step() {  # step <маркер> <описание> <команда…> — маркер только при коде 0
  local mk=$1 what=$2; shift 2
  [ -f "$mk" ] && { say "$what — уже готово"; return 0; }
  say "--- $what"
  if "$@"; then touch "$mk"; say "$what — готово"; else say "!!! $what — ошибки (см. runs/TURNIR/train_journal.csv) — стоп, повтор юнита продолжит"; exit 1; fi
}
step runs/TURNIR_E1_g1.done "группа 1 (mpk.train; Sohn — 5 затравок в M13)" python scripts/turnir_train.py run --groups 1 --jobs "$J" --jobs-small "$JS"
# ГП → M13 (0078: после группы 1 — M13 по предрегистрации; m13_chain_v3.sh ждёт runs/TURNIR_E1_g1.done). Группе 3 нужен
# Sohn, а группа 2 длинная (≈ 195 ч) — поэтому сначала M13 (≈ 22 ч), ГП не простаивает: M13 стартует сам после маркера.
say "жду M13 (runs/UzSohn_M13.done; юнит m13-sohn-2)"
while ! m13_done; do sleep 120; done
CUDA_VISIBLE_DEVICES= nice python scripts/turnir_eval.py stoprule >> runs/TURNIR/stoprule.log 2>&1 && say "правило остановки: $(grep -m1 'Δ =' runs/TURNIR/stoprule.md)" \
  || say "!!! правило остановки не посчитано (runs/TURNIR/stoprule.log) — продолжаю Э1"
# группа 3 — Sohn в общем рецепте: после порта в build_model (0063 п. 3); пока порта нет, план её не запускает (без маркера)
python scripts/turnir_train.py plan --groups 3 | grep -q "нет порта Sohn" && say "группа 3: порта Sohn в build_model нет — пропуск (повтор юнита после порта)" \
  || step runs/TURNIR_E1_g3.done "группа 3 (Sohn, общий рецепт)" python scripts/turnir_train.py run --groups 3 --jobs "$J" --jobs-small "$JS"

# честная M10-1b (0063): сделана отдельным юнитом m10-1bh (runs/M10_1bh_train.done) — здесь только пропуск/досчёт
if [ ! -f runs/M10_1bh_train.done ]; then
  [ -d configs/M10_1bh ] || python scripts/m10_1b_honest.py prepare
  python scripts/m10_1b_honest.py run --jobs 1 || { say "!!! M10-1b честная — ошибки, стоп"; exit 1; }
fi

step runs/TURNIR_E1_g2.done "группа 2 (родные рецепты SL-GCN/ST-GCN их кодом)" python scripts/turnir_train.py run --groups 2 --jobs "$J" --jobs-small "$JS"

wait
say "вероятности (ЦП)"
CUDA_VISIBLE_DEVICES= python scripts/turnir_eval.py export >> runs/TURNIR/export.log 2>&1 || { say "!!! выгрузка упала (runs/TURNIR/export.log)"; exit 1; }
python scripts/turnir_eval.py check || { say "!!! сверка стенда не пройдена — стоп"; exit 1; }
python scripts/turnir_eval.py summary && python scripts/turnir_eval.py inventory || { say "!!! сводка упала"; exit 1; }
touch runs/TURNIR_E1.done
say "=== Э1 готов: runs/TURNIR/turnir_summary.csv, runs/TURNIR/pairs.md (числа корпуса — предварительно до проверки меток)"
say "сеанс стоимости — отдельно, на простаивающей машине: CUDA_VISIBLE_DEVICES= python scripts/turnir_cost.py"
