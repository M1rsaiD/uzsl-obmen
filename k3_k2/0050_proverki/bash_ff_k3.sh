#!/usr/bin/env bash
# к3 0050: при (а) bash цепочки турнира (turnir_e1_chain.sh f514084 — без функции main, bash читает файл по ходу) жив во время
# слияния инструментов и перемотки Б10. Что будет, если git заменит файл под идущим bash? Сравнение: git merge --ff-only против правки на месте.
set -u; D=$(mktemp -d); cd "$D"; git init -q -b master .; git config user.email k3@x; git config user.name k3
v1='#!/usr/bin/env bash\necho A1\nsleep 3\necho A2-staroe\necho A3-staroe\n'; v2='#!/usr/bin/env bash\necho A1\nsleep 3\necho XXXXXXXXXXXXXXXXXXXX-novoe-dlinnee\necho B3-novoe\n'
printf "$v1" > c.sh; git add c.sh; git commit -qm v1; git checkout -qb nov; printf "$v2" > c.sh; git commit -qam v2; git checkout -q master
echo "git $(git --version | cut -d' ' -f3), bash $BASH_VERSION"
i0=$(stat -c %i c.sh); bash c.sh > o1.txt 2>&1 & sleep 1; git merge -q --ff-only nov; i1=$(stat -c %i c.sh); wait
echo "1) git merge --ff-only во время sleep (inode $i0 → $i1):"; sed 's/^/   /' o1.txt
git reset -q --hard HEAD~1; bash c.sh > o2.txt 2>&1 & sleep 1; printf "$v2" > c.sh; wait
echo "2) запись в тот же файл (тот же inode, как редактор/printf >):"; sed 's/^/   /' o2.txt
rm -rf "$D"
