#!/usr/bin/env bash
# к3 0055: цепочка Э2 53a0988, строки 239–251 — пустой список фолдов корпуса (python строки 239 упал) → need … 0 проходит?
# Функции batch/need (строки 189–205) и хвост 239–252 берутся из файла дословно (sed), run — заглушка: пишет metrics.json
# при непустом теге; local tag=$2 — как в run цепочки (строка 178), set -u — как строка 39.
set -u
CH=${1:?путь к turnir_e2_chain.sh}
W=$(mktemp -d); cd "$W"; mkdir -p runs/UzSohn
say() { echo "[say] $*"; }
run() { local tag=$2 f=$4 s=$5
  [ -n "$tag" ] || { say "!!! ОШИБКА  ${f:-} seed${s:-} (пустая строка задания)"; return 0; }
  mkdir -p runs/UzSohn/$tag/$f/seed$s && echo '{}' > runs/UzSohn/$tag/$f/seed$s/metrics.json; }
python() { case "$*" in *uzsl_loso.json*) echo "Traceback … FileNotFoundError (заглушка)" >&2; return 1;; *) command python3 "$@";; esac; }
J=2; CUR=u01b; SEEDS="42 43 44 45 46"; FOLDS5="fold1 fold2 fold3 fold4 fold5"
sleep() { :; }
eval "$(sed -n '189,205p' "$CH")"
echo "строки 189–205: $(sed -n '189p;201p' "$CH" | awk '{print $1}' | paste -sd' ')"
eval "$(sed -n '239,252p' "$CH" | sed 's/^/  /' | sed '1s/^/f() {\n/;$s/$/\n}/')"
( f ); rc=$?
echo "хвост цепочки (239–252) завершился с кодом $rc"
echo "uzsohn_e1_uzsl_pre: $(ls runs/UzSohn/uzsohn_e1_uzsl_pre 2>/dev/null | wc -l) каталогов; UzSohn_E2_E1.done: $([ -f runs/UzSohn_E2_E1.done ] && echo ЕСТЬ || echo нет)"
