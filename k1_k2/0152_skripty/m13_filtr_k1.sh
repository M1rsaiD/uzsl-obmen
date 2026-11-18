#!/usr/bin/env bash
# Второй путь теста, к1: что показывает фильтр журнала M13 «grep -E '===|!!!'» (0161 разд. 3) — по m13_chain_v4.sh (блоб 7dd40f7)
# и m13_train.py (блоб 8ac5f0f, восстановлен в 0148). Запуск: bash m13_filtr_k1.sh <m13_chain_v4.sh> <m13_train.py>
set -u
for f in "$1" "$2"; do echo "$(basename $f): блоб $(git hash-object "$f" | cut -c1-7)"; done
echo "--- строки цепочки, которые пропускает фильтр:"; grep -n -E '===|!!!' "$1" | cut -c1-150
echo "--- откуда в журнале числа теста (фильтр их не пропускает):"; grep -n -E 'tail -1 \$log|tables.out$' "$1" | cut -c1-150
echo "--- m13_train.py: строк с '===' или '!!!': $(grep -c -E '===|!!!' "$2")"
