#!/usr/bin/env bash
# Проба C: SIGKILL всем процессам юнита Э2 (как systemd-oomd / systemctl kill -s KILL) — trap не сработает, подъём только через ExecStopPost=
set -u
P=/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/probe_systemd/real
cd $P; : > runs/turnir_e2_chain.log
log(){ echo "[$(date +%H:%M:%S)] $*"; }
for u in k2-probe-e2c k2-probe-e1; do systemctl --user stop $u 2>/dev/null; systemctl --user reset-failed $u 2>/dev/null; done
systemd-run --user --unit=k2-probe-e2c --collect -p WorkingDirectory=$PWD -p MemoryMax=40G \
  -p MemorySwapMax=0 -p OOMPolicy=continue -p ExecStopPost='-/usr/bin/bash scripts/turnir_e2_chain.sh turnir_up' \
  -E PATH="$PATH" -E HOME="$HOME" -E E2_MODE=ii -E TURNIR_E1_UNIT=k2-probe-e1 -E TURNIR_E1_CMD="sleep 600" \
  bash -c "bash scripts/turnir_e2_chain.sh >> runs/turnir_e2_chain.log 2>&1"
sleep 8; log "C1 e2c: $(systemctl --user is-active k2-probe-e2c); процессов в юните: $(systemctl --user show -p TasksCurrent --value k2-probe-e2c); e1: $(systemctl --user is-active k2-probe-e1)"
systemctl --user kill -s KILL k2-probe-e2c; sleep 6
log "C2 после SIGKILL всем: e2c: $(systemctl --user is-active k2-probe-e2c); e1: $(systemctl --user is-active k2-probe-e1); WorkingDirectory e1: $(systemctl --user show -p WorkingDirectory --value k2-probe-e1)"
log "== журнал пробы:"; sed 's/^/    /' runs/turnir_e2_chain.log
for u in k2-probe-e2c k2-probe-e1; do systemctl --user stop $u 2>/dev/null; systemctl --user reset-failed $u 2>/dev/null; done; sleep 1
log "уборка: $(systemctl --user list-units --all --no-legend 'k2-probe-*' | wc -l) юнитов k2-probe-* осталось"
