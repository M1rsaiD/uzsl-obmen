#!/usr/bin/env bash
# Б10, к1: блок перемотки — дословно строки 11–19 диффа 0168_e2_tools_b10_diff_ot_61f1204.diff (0b22f9b) — на 12 git-случаях;
# плюс опыт: файл исполняемого bash-скрипта меняет git merge --ff-only (новый inode) против перезаписи на месте.
# Запуск: bash b10_k1.sh <путь к диффу>
set -u
D=$(realpath "$1"); S=$(mktemp -d); cd "$S"
sed -n '11,19p' "$D" | sed 's/^+//' > blok.sh
cat > run.sh <<'E2'
say() { echo "[say] $*"; }
cd "$1"; source "$2"
echo "HEAD=$(git rev-parse --short HEAD) ветка=$(git symbolic-ref -q --short HEAD || echo detached) OK=$([ -f runs/TURNIR_E2_OK ] && cut -c1-7 runs/TURNIR_E2_OK || echo нет)"
E2
mk() {  # master = база, lev = база + рычаги; PREOK = lev
  r=$S/$1; git init -q -b master $r; cd $r; git config user.email t@t; git config user.name t
  mkdir -p src runs scripts; echo runs/ > .gitignore; echo base > src/sohn.py; echo chain > scripts/turnir_e2_chain.sh
  git add -A; git commit -qm base; git checkout -q -b lev; echo levers >> src/sohn.py; git commit -qam lev
  git rev-parse HEAD > runs/TURNIR_E2_PREOK; git checkout -q master; cd $S; }
case_run() { echo "== $1"; bash run.sh $S/$1 $S/blok.sh 2>&1 | sed "s/$(cd $S/$1; git rev-parse lev)/<lev>/g"; }
mk ff;        case_run ff
mk notff;     (cd notff; echo x > notes.txt; git add notes.txt; git commit -qm moved); case_run notff
mk dirty;     (cd dirty; echo y >> .gitignore); case_run dirty
mk staged;    (cd staged; echo z > new.txt; git add new.txt); case_run staged
mk detached;  (cd detached; git checkout -q --detach); case_run detached
mk ahead;     (cd ahead; git merge -q --ff-only lev; echo w > a.txt; git add a.txt; git commit -qm ahead); case_run ahead
mk same;      (cd same; git merge -q --ff-only lev); case_run same
mk okexists;  (cd okexists; echo manual > runs/TURNIR_E2_OK); case_run okexists
mk nopreok;   rm nopreok/runs/TURNIR_E2_PREOK; case_run nopreok
mk badhash;   echo "deadbeef$(printf '%032d' 0)" > badhash/runs/TURNIR_E2_PREOK; case_run badhash
mk short;     (cd short; git rev-parse --short lev > runs/TURNIR_E2_PREOK); case_run short
mk untracked; (cd untracked; git checkout -q lev; echo f > src/new_lever.py; git add src/new_lever.py; git commit -qm lev2
               git rev-parse HEAD > runs/TURNIR_E2_PREOK; git checkout -q master; echo local > src/new_lever.py); case_run untracked
echo "== опыт: исполняемый скрипт меняется git merge --ff-only"
git init -q -b master self; cd self; git config user.email t@t; git config user.name t
printf '%s\n' 'echo "старая: шаг 1"; i0=$(stat -c %i chain.sh)' 'git merge --ff-only -q lev' \
  'echo "старая: шаг 2 (inode $i0 → $(stat -c %i chain.sh))"' 'echo "старая: шаг 3"' > chain.sh
git add chain.sh; git commit -qm v1; git checkout -q -b lev
printf '%s\n' 'echo "НОВАЯ: шаг 1"' 'git merge --ff-only -q lev' 'echo "НОВАЯ: шаг 2"' 'echo "НОВАЯ: шаг 3"; echo "НОВАЯ: шаг 4"' > chain.sh
git commit -qam v2; git checkout -q master; bash chain.sh; echo "файл после: $(head -1 chain.sh | cut -c1-30)"
echo "== контроль: тот же скрипт, но замена перезаписью на месте (>)"
sed -i 's/git merge --ff-only -q lev/git show lev:chain.sh > chain.sh/' chain.sh; git commit -qam v1b; bash chain.sh 2>&1 | head -3
cd /; rm -rf "$S"
