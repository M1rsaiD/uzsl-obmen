#!/usr/bin/env bash
# Ждёт новой строки в журнале OBMEN.md (без расхода токенов).
# По умолчанию — писем консультанту_2 от консультантов 1 и 3 («к1 → к2|к3 → к2»); консультант_1 ждёт с аргументом «к2 → к1»,
# консультант_3 — «к2 → к3». Несколько направлений — через «|».
# Выходит с кодом 0 и печатает новые строки; с кодом 1 — по таймауту.
# Запуск (в фоне): tools/zhdat_otveta.sh [интервал_с=180] [таймаут_мин=180] [направления="к1 → к2|к3 → к2"]
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
f=OBMEN.md
interval=${1:-180}
limit_min=${2:-180}
dir=${3:-к1 → к2|к3 → к2}
# «X → Y» ловит и письма нескольким адресатам: «X → Y, к3», «X → к3, Y» (раньше пропускались)
re="\| ($(printf '%s' "$dir" | sed -E 's/(к[0-9]) → (к[0-9])/\1 → (к[0-9], )*\2(, к[0-9])*/g')) \|"
# считаем по origin/main после fetch: правки в рабочей копии не мешают (git pull --rebase с ними падал)
count() { git show "origin/main:$f" 2>/dev/null | grep -cE "$re"; }
before=$(count)
start=$(date +%s)
while :; do
  git fetch -q origin >/dev/null 2>&1 || echo "git fetch не прошёл, повторю" >&2
  now=$(count)
  if [ "$now" -gt "$before" ]; then
    git rebase -q origin/main >/dev/null 2>&1 || echo "рабочая копия не обновлена — сделайте: git rebase origin/main" >&2
    echo "Новое ($dir):"
    git show "origin/main:$f" | grep -E "$re" | tail -n $((now - before))
    exit 0
  fi
  if [ $(( $(date +%s) - start )) -ge $(( limit_min * 60 )) ]; then
    echo "Нового ($dir) нет за $limit_min мин." >&2
    exit 1
  fi
  sleep "$interval"
done
