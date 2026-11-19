#!/usr/bin/env bash
# к1 0156: m13_train.py 49b93bd (блоб 5f83fec) = 8ac5f0f (сборка к1 0148) + хунки m13_train из диффа 0167; сверка блоба
# запуск: bash rec_5f83fec.sh <каталог с scripts/m13_train.py блоба 8ac5f0f> (из корня uzsl-obmen)
set -eu
D=$1
[ "$(git hash-object $D/scripts/m13_train.py)" = 8ac5f0faa29b4b9d7226845d8f49835e674dcec4 ]
sed -n '1,154p' k2_k1/0167_e2_infra_levers_diff_ot_d7b0d68.diff | (cd $D && patch -p1 -s)
git hash-object $D/scripts/m13_train.py   # ожидается 5f83fecd2fcbccc582603fcc14ee32fc80f58be2
