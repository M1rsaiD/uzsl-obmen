#!/usr/bin/env bash
# 4.7 (часть), к1: цепочка Э2 a1deebc — восстановить из диффов обмена (0151_skripty/rec_964ecba.sh + 0171), сверить блоб 53a0988,
# bash -n, затем исполнить ожидание свободной ГП с PID-логом (строки 120–127 a1deebc) с заглушками gpu_busy / nvidia-smi / sleep / say.
# Запуск из корня uzsl-obmen: bash k1_k2/0153_skripty/a1deebc_k1.sh
set -eu
T=$(mktemp -d); bash k1_k2/0151_skripty/rec_964ecba.sh "$T" > /dev/null
awk '/^diff --git /{p=index($0, "scripts/turnir_e2_chain.sh")>0} p' k2_k1/0171_e2_tools_diff_964ecba_a1deebc.diff > "$T/d.diff"
(cd "$T" && patch -s -p1 < d.diff)
h=$(git hash-object "$T/scripts/turnir_e2_chain.sh"); echo "блоб цепочки: ${h:0:7} (ожидается 53a0988)"; [ "${h:0:7}" = 53a0988 ]
bash -n "$T/scripts/turnir_e2_chain.sh" && echo "bash -n: ok"
echo "переменная n в цепочке: $(grep -c -E '(^|[^A-Za-z_])n=' "$T/scripts/turnir_e2_chain.sh") присваивания (121: n=0; 123: n+1), других нет"
{ echo 'set -u; c=0; sleep() { :; }; say() { echo "[say] $*"; }'
  echo 'gpu_busy() { c=$((c + 1)); [ $c -le 130 ] || [ $c -eq 135 ]; }   # занято 130 мин, свободно, на 5-й мин окна — снова занято'
  echo 'nvidia-smi() { printf "4242, python\n77, Xorg\n4343, python\n"; }'
  sed -n '120,127p' "$T/scripts/turnir_e2_chain.sh"
  echo 'echo "вышел: проверок gpu_busy $c, минут занятости n=$n"'; } > "$T/frag.sh"
bash "$T/frag.sh"
rm -rf "$T"
