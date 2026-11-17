#!/usr/bin/env bash
# Сухая проба 22.1 на НАСТОЯЩЕМ turnir_e2_chain.sh (копия бит в бит, ветка turnir-e2-tools 61f1204) в каталоге пробы:
# юниты k2-probe-e2* / k2-probe-e1 вместо turnir-e2 / turnir-e1, подъём — пустышка (TURNIR_E1_CMD); настоящие юниты не трогаются
set -u
P=/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/probe_systemd/real
cd $P; : > runs/turnir_e2_chain.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
clean(){ for u in k2-probe-e2a k2-probe-e2b k2-probe-e1; do systemctl --user stop $u 2>/dev/null; systemctl --user reset-failed $u 2>/dev/null; done; sleep 1; }
e2(){  # та же команда, что в шапке цепочки (стр. 22–24), другое имя + переменные пробы
  systemd-run --user --unit=$1 --collect -p WorkingDirectory=$PWD -p MemoryMax=40G \
    -p MemorySwapMax=0 -p OOMPolicy=continue -p ExecStopPost='-/usr/bin/bash scripts/turnir_e2_chain.sh turnir_up' \
    -E PATH="$PATH" -E HOME="$HOME" -E E2_MODE=ii -E TURNIR_E1_UNIT=k2-probe-e1 -E TURNIR_E1_CMD="sleep 600" \
    bash -c "bash scripts/turnir_e2_chain.sh >> runs/turnir_e2_chain.log 2>&1"; }
props(){ systemctl --user show -p ActiveState -p WorkingDirectory -p MemoryMax -p MemorySwapMax -p Environment k2-probe-e1 | sed 's/^/    /'; }
clean
log "== A. kill -9 главного процесса юнита Э2 (SIGKILL: trap не сработает — только ExecStopPost=)"
e2 k2-probe-e2a; sleep 8
log "A1 e2a: $(systemctl --user is-active k2-probe-e2a); OOMPolicy/ManagedOOM: $(systemctl --user show -p OOMPolicy -p ManagedOOMMemoryPressure --value k2-probe-e2a | paste -sd' '); e1 до: $(systemctl --user is-active k2-probe-e1)"
pid=$(systemctl --user show -p MainPID --value k2-probe-e2a); log "A2 MainPID=$pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-90)"
kill -9 $pid; sleep 6
log "A3 e2a: $(systemctl --user is-active k2-probe-e2a); e1: $(systemctl --user is-active k2-probe-e1)"; props
log "A4 процессы цепочки в каталоге пробы после kill: $(pgrep -fc 'bash scripts/turnir_e2_chain.sh$')"
clean
log "== B. обычный stop юнита Э2 (SIGTERM → trap exit 143 → trap EXIT turnir_up; затем ExecStopPost= — охрана is-active)"
e2 k2-probe-e2b; sleep 8
systemctl --user stop k2-probe-e2b; sleep 4
log "B1 e2b: $(systemctl --user is-active k2-probe-e2b); e1: $(systemctl --user is-active k2-probe-e1); экземпляров k2-probe-e1: $(systemctl --user list-units --all --no-legend 'k2-probe-e1*' | wc -l)"; props
log "== журнал цепочки пробы (runs/turnir_e2_chain.log):"; sed 's/^/    /' runs/turnir_e2_chain.log
clean
log "уборка: $(systemctl --user list-units --all --no-legend 'k2-probe-*' | wc -l) юнитов k2-probe-* осталось"
