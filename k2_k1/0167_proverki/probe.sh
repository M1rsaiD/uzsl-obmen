#!/usr/bin/env bash
# Сухая проба механизма (ii′) на пустышках: имена k2-probe-*, turnir-e1 не трогается
set -u
P=/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/probe_systemd
cd $P; : > up_stub.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
for u in a b c d f g; do systemctl --user stop k2-probe-$u 2>/dev/null; systemctl --user reset-failed k2-probe-$u 2>/dev/null; done
log "== 1. второй systemd-run с занятым именем"
systemd-run --user --unit=k2-probe-a --collect sleep 300 2>&1; log "1a rc=$?"
systemd-run --user --unit=k2-probe-a --collect sleep 300 2>&1; log "1b то же имя: rc=$?"
systemctl --user stop k2-probe-a
log "== 2. is-active, пока юнит в activating"
systemd-run --user --unit=k2-probe-b --collect -p Type=oneshot --no-block sleep 30 2>&1; log "2a rc=$?"
sleep 1; st=$(systemctl --user is-active k2-probe-b); rc=$?; log "2b is-active=$st rc=$rc"
systemd-run --user --unit=k2-probe-b --collect sleep 300 2>&1; log "2c то же имя в activating: rc=$?"
systemctl --user stop k2-probe-b
log "== 3. ExecStopPost=- (ошибка не роняет юнит) против без «-»"
systemd-run --user --unit=k2-probe-c -p ExecStopPost=-/bin/false sleep 300 >/dev/null 2>&1
systemd-run --user --unit=k2-probe-d -p ExecStopPost=/bin/false sleep 300 >/dev/null 2>&1
systemctl --user stop k2-probe-c k2-probe-d; sleep 1
log "3 с «-»: $(systemctl --user is-failed k2-probe-c); без «-»: $(systemctl --user is-failed k2-probe-d)"
systemctl --user reset-failed k2-probe-c k2-probe-d 2>/dev/null
log "== 4. kill -9 главного процесса → ExecStopPost → подъём второго юнита изнутри юнита"
systemd-run --user --unit=k2-probe-f --collect -p WorkingDirectory=$P -p MemoryMax=40G -p MemorySwapMax=0 -E PATH="$PATH" -E HOME="$HOME" -E UP_UNIT=k2-probe-g -p "ExecStopPost=-/usr/bin/bash up_stub.sh" sleep 600 2>&1
sleep 1; pid=$(systemctl --user show -p MainPID --value k2-probe-f); log "4a f активен: $(systemctl --user is-active k2-probe-f), MainPID=$pid"
kill -9 $pid; sleep 4
log "4b f: $(systemctl --user is-active k2-probe-f); g: $(systemctl --user is-active k2-probe-g); g WorkingDirectory=$(systemctl --user show -p WorkingDirectory --value k2-probe-g)"
log "4c g Environment: $(systemctl --user show -p Environment --value k2-probe-g | tr ' ' '\n' | grep -E '^TURNIR_' | paste -sd' ')"
log "== 5. повторный вызов подъёма при живом g — второго экземпляра нет"
UP_UNIT=k2-probe-g bash up_stub.sh; log "5 g: $(systemctl --user is-active k2-probe-g); юнитов k2-probe-g: $(systemctl --user list-units --all --no-legend 'k2-probe-g*' | wc -l)"
log "== журнал пустышки:"; cat up_stub.log
systemctl --user stop k2-probe-g 2>/dev/null; sleep 1
log "уборка: $(systemctl --user list-units --all --no-legend 'k2-probe-*' | wc -l) юнитов k2-probe-* осталось"
