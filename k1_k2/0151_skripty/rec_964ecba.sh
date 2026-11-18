#!/usr/bin/env bash
# Восстановить scripts/turnir_eval.py и scripts/turnir_e2_chain.sh ветки turnir-e2-tools на 964ecba из файлов обмена.
# turnir_eval.py (блоб 0789dc5):
# master f6a5a72 (c4e557d; строки 42–43 замаскированы к2) + хунки turnir_eval.py из диффов 0167 (→ 46a9d1d) и 0169 (→ 0789dc5).
# Запуск из корня uzsl-obmen: bash k1_k2/0151_skripty/rec_964ecba.sh <каталог>
set -eu
O=$(realpath "$1"); mkdir -p "$O/scripts"
cp k2_k1/0168_fajly/turnir_eval_master_f6a5a72_strk42-43_maskirovany.py "$O/scripts/turnir_eval.py"
x() { awk '/^diff --git /{p=($0 ~ "scripts/turnir_eval.py")} p' "$1"; }
x k2_k1/0167_e2_tools_diff_ot_f6a5a72.diff > "$O/p1.diff"; x k2_k1/0169_e2_tools_diff_0b22f9b_964ecba.diff > "$O/p2.diff"
(cd "$O" && patch -s -p1 < p1.diff && patch -s -p1 < p2.diff)
# turnir_e2_chain.sh: новый файл в 0167 (792c47e) → 0168 (2238feb) → 0169 (1320b43); блоб сверяется на каждом шаге
y() { awk '/^diff --git /{p=index($0, "scripts/turnir_e2_chain.sh")>0} p' "$1"; }
n=0; for d in k2_k1/0167_e2_tools_diff_ot_f6a5a72.diff k2_k1/0168_e2_tools_b10_diff_ot_61f1204.diff k2_k1/0169_e2_tools_diff_0b22f9b_964ecba.diff; do
  n=$((n + 1)); y "$d" > "$O/c$n.diff"; (cd "$O" && patch -s -p1 < c$n.diff)
  h=$(git hash-object "$O/scripts/turnir_e2_chain.sh"); w=$(sed -n '1,3p' "$O/c$n.diff" | grep -oE '\.\.[0-9a-f]{7}' | cut -c3-)
  [ "${h:0:7}" = "$w" ] || { echo "!!! блоб цепочки $h ≠ $w (шаг $n)"; exit 1; }
done
echo "восстановлен: $O/scripts/turnir_e2_chain.sh (блоб $(git hash-object "$O/scripts/turnir_e2_chain.sh") — как в 0169)"
echo "восстановлен: $O/scripts/turnir_eval.py ($(wc -l < "$O/scripts/turnir_eval.py") строк; хунки легли без смещений и отказов)"
