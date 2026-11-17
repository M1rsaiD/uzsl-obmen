#!/usr/bin/env bash
# Б11, к1: строки 18, 20 и 49–52 turnir_e1_chain.sh (f514084) — дословно из файла; python — заглушка в PATH (правило считается 3 с).
# Запуск: bash b11_k1.sh <путь к turnir_e1_chain.sh>
set -u
F=$(realpath "$1"); W=$(mktemp -d); cd "$W"; mkdir -p runs/TURNIR
{ sed -n '18p;20p' "$F"; sed -n '49,52p' "$F"; } > frag.sh
mkdir bin; printf '#!/bin/sh\nsleep 3; echo "Δ = +1.00" > runs/TURNIR/stoprule.md\n' > bin/python; chmod +x bin/python
export PATH="$W/bin:$PATH"   # строка 51 зовёт python через nice — нужен исполняемый файл, не функция
echo "строка 51 = $(sed -n '51p' "$F" | cut -c1-90)…"
# случай 1: M13 кончается через 1 с после старта ожидания (окно «между проверкой и stop»)
( sleep 1; touch runs/UzSohn_M13.done ) &
sed -i 's/sleep 120/sleep 0.2/' frag.sh
bash frag.sh > e1.log 2>&1 & pid=$!
sleep 2.2; echo "идёт правило остановки (pid жив: $(kill -0 $pid 2>/dev/null && echo да || echo нет)); tail -1 журнала: $(tail -1 e1.log | cut -c23-)"
wait $pid; echo "после правила tail -1: $(tail -1 e1.log | cut -c23-)"
# случай 2: лежит старый runs/M13.done, UzSohn_M13.done нет — цикл строки 50 не ждёт вовсе
rm -f runs/UzSohn_M13.done runs/TURNIR/stoprule.md e1.log; touch runs/M13.done
timeout 10 bash frag.sh > e1.log 2>&1; echo "с runs/M13.done и без UzSohn_M13.done: $(grep -c 'правило остановки' e1.log) строка правила — цикл не ждал"
rm -rf "$W"
