#!/usr/bin/env bash
# к1 0157: turnir_val_m13.py из диффов обмена 0172 → 0174 → 0176, блоб на каждом шаге (из корня uzsl-obmen; без сети)
set -eu
W=$(mktemp -d); O=$PWD
cut1() { awk '/^diff --git a\/scripts\/turnir_val_m13.py/{p=1} /^diff --git/ && !/turnir_val_m13/{p=0} p' "$O/$1"; }
cd $W && git init -q .
for d in k2_k1/0172_b9_diff_a1deebc_e5205e4.diff k2_k1/0174_b9_diff_e5205e4_84eaf0b.diff k2_k1/0176_port_diff_e516900_c9a9c2a.diff; do
  cut1 $d | git apply; echo "$d → $(git hash-object scripts/turnir_val_m13.py)"
done   # ожидается 7718975, c6186af, 93433e3
rm -rf $W
