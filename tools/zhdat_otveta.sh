#!/usr/bin/env bash
# Ждёт ответа консультанта_1 в репозитории (без расхода токенов).
# Выходит с кодом 0, когда в OBMEN.md появилась новая строка «к1 → к2»,
# и печатает её; с кодом 1 — по таймауту.
# Запуск (в фоне, из любой папки репозитория): tools/zhdat_otveta.sh [интервал_с=180] [таймаут_мин=180]
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
f=OBMEN.md
interval=${1:-180}
limit_min=${2:-180}
count() { grep -c '| к1 → к2 |' "$f"; }
before=$(count)
start=$(date +%s)
while :; do
  git pull -q --rebase >/dev/null 2>&1 || echo "git pull не прошёл, повторю" >&2
  now=$(count)
  if [ "$now" -gt "$before" ]; then
    echo "Ответ консультанта_1:"
    grep '| к1 → к2 |' "$f" | tail -n $((now - before))
    exit 0
  fi
  if [ $(( $(date +%s) - start )) -ge $(( limit_min * 60 )) ]; then
    echo "Ответа нет за $limit_min мин — сообщи автору." >&2
    exit 1
  fi
  sleep "$interval"
done
