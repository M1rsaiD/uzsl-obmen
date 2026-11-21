#!/usr/bin/env bash
# к1 0159: что из строк PREOK (к2 0180) я могу сверить по обмену сам. Запуск из корня uzsl-obmen; без сети.
set -eu
W=$(mktemp -d); trap 'rm -rf $W' EXIT
cut1() { awk -v f="$2" '$0 ~ "^diff --git a/"f" " {p=1; print; next} /^diff --git/ {p=0} p' "$1"; }
echo "## 1. скрипт порта из диффов 0175 (новый файл) и 0176 — ожидается fc5d15a, затем ae84ddb (ls-tree PH)"
(cd $W && git init -q . && for d in 0175_g3_port_diff_84eaf0b_e516900 0176_port_diff_e516900_c9a9c2a; do
   cut1 "$OLDPWD/k2_k1/$d.diff" scripts/turnir_sohn_shared.py | git apply; git hash-object scripts/turnir_sohn_shared.py; done)
echo "## 2. файлы рычагов f6a5a72 → fa5e37c (объединение файлов диффов 0163, 0163 B, 0167 рычаги, 0171) — должно = name-only master..PH"
for d in 0163_e2_levers_diff_ot_f6a5a72 0163_n4_variant_B 0167_e2_infra_levers_diff_ot_d7b0d68 0171_e2_levers_diff_49b93bd_fa5e37c; do
  grep '^diff --git' k2_k1/$d.diff | awk '{print substr($3, 3)}'; done | sort -u
echo "## 3. индексы блобов по диффам (файлов целиком у меня нет)"
grep -h -A1 '^diff --git a/scripts/uzsohn_code.txt' k2_k1/0163_e2_levers_diff_ot_f6a5a72.diff k2_k1/0167_e2_infra_levers_diff_ot_d7b0d68.diff | grep index
grep -A1 '^diff --git a/reports/TURNIR_prereg.md' k2_k1/0177_prereg_dopolnenie_254a82b.diff | grep index
echo "## 4. присланные файлы (0176_fajly) — ожидается d47d0c6, b7532dc"
git hash-object k2_k1/0176_fajly/uzsohn_sel_49b93bd.py k2_k1/0176_fajly/uzsohn_pose_49b93bd.py
