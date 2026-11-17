#!/usr/bin/env bash
# Скептик ред. 3 поправки 4 (линза — верность согласованиям): повтор ключевых проверок. Только чтение.
S=/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad
R3=$S/letter/0165/TURNIR_prereg_popravka4_DRAFT.md
W=$S/wt_e2
OM=~/uzsign/osnovnaya_model
echo "== H1: turnir-e1 — транзиентный юнит (systemd-run --collect), рестарт в ред. 3 — systemctl start"
grep -E '^# This is a transient|CollectMode' /run/user/1000/systemd/transient/turnir-e1.service
grep -o -E 'TURNIR_JOBS[A-Z_]*=[0-9]+' /run/user/1000/systemd/transient/turnir-e1.service
grep -n 'J=${TURNIR_JOBS' $OM/scripts/turnir_e1_chain.sh
grep -n -E 'systemd-run --user --unit=turnir-e1' $OM/scripts/turnir_e1_chain.sh
grep -n -E 'systemd-run --user --unit=m13-sohn-2' $OM/scripts/uzsohn_stage2_chain.sh
grep -n -E 'systemctl --user start turnir-e1' $R3 | cut -c1-120
echo "== M1: цена отступления (ii′) через 12 ч"
grep -n -E 'та же цена|≈ 195 ч позже|1–2 суток позже|при \(ii′\) — после группы 2' $R3 | cut -c1-140
grep -n 'затем (б)' $S/letter/k3_0043/0043.md | cut -c1-120
echo "== M2: корпус без --input против 22.8 (UzSohn в Э1 на корпусе)"
grep -n -E 'корпус с переносом 11 × 5|по-прежнему запрещён' $R3 | cut -c1-140
grep -n -E '^POSE_NPZ|^ASPECT_NPZ|a.set in \("sb", "alluz", "slovo", "bukva"\)' $W/scripts/m13_train.py | cut -c1-140
echo "== M3: время пробы памяти ред. 2 → ред. 3"
grep -n 'проба памяти' $S/letter/0159/TURNIR_prereg_popravka4_DRAFT.md $R3 | cut -c1-160
echo "== П29: sohn_groups и порядок блоков"
git -C $OM show f6a5a72:scripts/m13_train.py | sed -n '475,479p'
echo "== № 2: параметры n_tok · 37 440 + 37 248"
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 nice -n 19 /home/error_404/.local/share/mise/installs/python/3.14.7/bin/python -I -c "print({n: n*(192+192*192+192+192)+192*192+192+192 for n in (3,4)})"
echo "== фильтр M13"
grep -c -E '===|!!!' $OM/scripts/m13_tables.py $OM/scripts/m13_train.py
