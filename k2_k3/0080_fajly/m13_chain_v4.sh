#!/usr/bin/env bash
# M13-Sohn, v4 (16.11.2026; копия m13_chain_v3.sh, v3 не меняется): n_done считает только затравки цепочки ($SEEDS).
# В runs/UzSohn/sb/all лежит seed48 (контроль этапа 1 UzSohn, 12.11): с ним после 45 и 46 было бы 6 ≠ 5, и «need sb 5»
# остановил бы цепочку до корпуса (проверка к2 по к1 0136, разд. 4). Больше ничего не меняется; команда — ниже (v4).
# M13-Sohn, v3 (06.11.2026, решение к1 0078 «Э0 не пройден — продолжать с пометкой»; копия m13_chain_v2.sh, v2 не меняется):
# Э0 не повторяется (5 прогонов asl_e0 есть; итог «не пройден» — reports/m13_e0_check.json, runs/UzSohn_E0_FAIL), дальше —
# как в предрегистрации: Bukva → Slovo → предобучение → корпус ×2 → цена. Старт — после группы 1 турнира
# (runs/TURNIR_E1_g1.done) и свободной ГП 3 мин; перед каждым новым прогоном m13_train.py сам вызывает weekend_gate.sh.
#   cd ~/uzsign/osnovnaya_model && systemd-run --user --unit=m13-sohn-2 --collect -p WorkingDirectory=$PWD -p MemoryMax=40G \
#     -p MemorySwapMax=0 -E PATH="$PATH" -E HOME="$HOME" bash -c "bash scripts/m13_chain_v4.sh >> runs/m13_chain.log 2>&1"
# --- далее заголовок v2 ---
# M13-Sohn, поправка 1 (reports/M13_sohn_prereg.md, разд. 9; заменяет m13_chain.sh — тот сохранён, юнитом не запускается):
# (заголовок v2) runs/UzSohn, 5 затравок, Э0 (asl-signs) ПЕРВЫМ — не пройдено → стоп без маркера. Кэши входа из полных точек Э0 (ЦП) → ждёт ГП: runs/M12_3S.done (конец очереди
# M12: m12-2 → m12-2n → m12-4 → m12-3p → m12-3s), неактивность m12-2, m12-2n, m12-4, m12-3p, m12-3s и нет python на ГП
# 10 мин подряд (ручной старт без маркера: touch runs/M13_GO — тогда только «ГП свободна 10 мин») →
# Э0 asl-signs (их fold 0, 83 эпохи из 300) → Bukva (5 × 5) → Slovo (5 × 5) → предобучение S + B (5) → корпус с переносом (11 × 5) → таблицы →
# корпус с нуля (11 × 5, справочно) → стоимость (машина свободна) → таблицы и отчёт → runs/M13.done (только если всё есть).
# Одновременно M13_JOBS (3) прогонов на одной ГП (модель мала, шаг ограничен запуском ядер). Повтор продолжает с места.
#   cd ~/uzsign/osnovnaya_model && systemd-run --user --unit=m13-sohn --collect -p WorkingDirectory=$PWD -p MemoryMax=40G \
#     -p MemorySwapMax=0 -E PATH="$PATH" -E HOME="$HOME" bash -c "bash scripts/m13_chain_v2.sh >> runs/m13_chain.log 2>&1"
export PATH=$HOME/.local/share/mise/installs/python/3.14.7/bin:$PATH  # урок 30: каталог 3.14.7, не ссылка 3.14 (её двигает omarchy update → mise up); голый python с 11.11 — системный без numpy/torch
python -c 'import numpy, torch' 2> /dev/null || { echo "!!! ОШИБКА: $(command -v python) без numpy/torch (урок 30, ERRORS_LESSONS.md) — стоп"; exit 1; }
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=src:../uzsign_e1/src:../uzsign_ai_v2/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
J=${M13_JOBS:-3}
EXTRA=""
SEEDS="42 43 44 45 46"
FOLDS5="fold1 fold2 fold3 fold4 fold5"
LOSO=$(python -c "import json;print(' '.join(sorted(json.load(open('../uzsign_ai_v2/splits/uzsl_loso.json'))['folds'])))")
R=runs/UzSohn
mkdir -p $R/logs
say() { echo "[$(date '+%F %T')] $*"; }
gpu_py() { [ "$(nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -c python)" -gt 0 ]; }
busy() {  # v3: ждём конца группы 1 турнира (или ручной runs/M13_GO) и свободной ГП
  [ -f runs/TURNIR_E1_g1.done ] || [ -f runs/M13_GO ] || return 0
  gpu_py
}
n_done() { local s c=0; for s in $SEEDS; do c=$((c + $(ls $R/$1/*/seed$s/metrics.json 2>/dev/null | wc -l))); done; echo $c; }  # v4: только затравки цепочки

say "=== M13-Sohn: кэши входа (ЦП)"
python scripts/m13_build.py bukva slovo uzsl asl || { say "!!! ОШИБКА кэш"; exit 1; }
say "v3: жду ГП (runs/TURNIR_E1_g1.done, python на ГП нет 3 мин подряд)"
while true; do
  while busy; do sleep 60; done
  ok=1; for i in $(seq 3); do sleep 60; busy && { ok=0; break; }; done
  [ $ok = 1 ] && break
done
say "ГП свободна; прогонов параллельно: $J"

run_set() {  # строки «набор фолд затравка тег [init]» со стандартного входа
  while read -r set fold seed tag init; do
    [ -z "$set" ] && continue
    [ -f $R/$tag/$fold/seed$seed/metrics.json ] && continue
    while [ "$(jobs -rp | wc -l)" -ge "$J" ]; do wait -n; done
    (
      log=$R/logs/${tag}_${fold}_seed$seed.log
      if python scripts/m13_train.py --set $set --fold $fold --seed $seed --tag $tag ${init:+--init $init} $EXTRA > $log 2>&1; then
        say "$tag $fold seed$seed: $(tail -1 $log)"
      else
        say "!!! ОШИБКА $tag $fold seed$seed (см. $log)"
      fi
    ) &
  done
  wait
}
need() { [ "$(n_done $1)" -eq "$2" ] || { say "!!! $1: готово $(n_done $1) из $2 — стоп (перезапуск юнита продолжит)"; exit 1; }; }
nl=$(echo $LOSO | wc -w)

need asl_e0 5   # v3: Э0 не повторяется — 5 прогонов есть
say "Э0 НЕ ПРОЙДЕН (0078: продолжать с пометкой; разница −2,71 п. п., reports/m13_e0_check.json) — далее по предрегистрации"
EXTRA=""
say "=== Bukva"
run_set <<< "$(for s in $SEEDS; do for f in $FOLDS5; do echo bukva $f $s bukva; done; done)"
need bukva 25
say "=== Slovo"
run_set <<< "$(for s in $SEEDS; do for f in $FOLDS5; do echo slovo $f $s slovo; done; done)"
need slovo 25
python scripts/m13_tables.py > $R/tables.out 2>&1 || say "!!! ОШИБКА таблицы (промежуточные)"
say "=== предобучение Slovo + Bukva"
run_set <<< "$(for s in $SEEDS; do echo sb all $s sb; done)"
need sb 5
say "=== корпус LOSO, с предобучением"
run_set <<< "$(for s in $SEEDS; do for f in $LOSO; do echo uzsl $f $s uzsl_pre $R/sb/all/seed$s/best.pt; done; done)"
need uzsl_pre $((nl * 5))
python scripts/m13_tables.py > $R/tables.out 2>&1 || say "!!! ОШИБКА таблицы (промежуточные)"
say "=== корпус LOSO, с нуля (справочно)"
run_set <<< "$(for s in $SEEDS; do for f in $LOSO; do echo uzsl $f $s uzsl_scratch; done; done)"
need uzsl_scratch $((nl * 5))
say "=== стоимость: жду простоя (до 2 ч)"
for i in $(seq 120); do
  python scripts/m13_cost.py > $R/cost.log 2>&1 && break
  sleep 60
done
[ -f reports/m13_cost.json ] || { say "машина не простаивала 2 ч — замер с --force (пометка в m13_cost_force.json)"; python scripts/m13_cost.py --force > $R/cost.log 2>&1; }
python scripts/m13_tables.py > $R/tables.out 2>&1 || { say "!!! ОШИБКА таблицы"; exit 1; }
sed "s/^/[$(date '+%F %T')] /" $R/tables.out
touch runs/UzSohn_M13.done
say "=== M13-Sohn завершена (reports/m13_tables.md, M13_report.md)"
