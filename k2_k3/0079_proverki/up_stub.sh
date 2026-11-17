# пустышка turnir_up: та же охрана и те же -p/-E, что в 22.1; имя юнита — из UP_UNIT
log(){ echo "[$(date +%H:%M:%S)] up_stub: $*" >> up_stub.log; }
log "запущен; cwd=$PWD; XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-нет}; DBUS=${DBUS_SESSION_BUS_ADDRESS:-нет}"
systemctl --user is-active --quiet "$UP_UNIT" && { log "$UP_UNIT уже активен — не поднимаю"; exit 0; }
systemd-run --user --unit="$UP_UNIT" --collect -p WorkingDirectory=$PWD -p MemoryMax=40G -p MemorySwapMax=0 -E PATH="$PATH" -E HOME="$HOME" -E TURNIR_JOBS=1 -E TURNIR_JOBS_SMALL=1 bash -c "sleep 600" >> up_stub.log 2>&1
log "systemd-run rc=$?"
