#!/usr/bin/env bash
# П25, к1: самопроверка стенда turnir_eval.py 964ecba (строки 1013–1022 — строка Э3 «Sohn×3 − Sohn» и два случая к3) на восстановленном
# коде. Замаскированные к2 строки FOLDS (имена фолдов корпуса) заменяю нейтральной заглушкой S01…S12 — самопроверка их не читает.
# Запуск из корня uzsl-obmen: bash k1_k2/0152_skripty/selftest_k1.sh   (нужны numpy, scipy; без torch и сети)
set -eu
T=$(mktemp -d); bash k1_k2/0151_skripty/rec_964ecba.sh "$T" > /dev/null
python3 - "$T/scripts/turnir_eval.py" <<'PY'
import sys
p = sys.argv[1]; L = open(p, encoding="utf-8").read().split("\n")
i = [k for k, s in enumerate(L) if s.startswith("# [строка 42 замаскирована")][0]
assert L[i + 1].startswith("# [строка 43 замаскирована")
L[i], L[i + 1] = '         "UzSL": [f"S{k:02d}" for k in range(1, 13)]}  # заглушка к1', ""
open(p, "w", encoding="utf-8").write("\n".join(L))
PY
(cd "$T" && python3 scripts/turnir_eval.py --selftest 2>&1 | grep -v "^fatal: not a git repository" | tail -4)
rm -rf "$T"
