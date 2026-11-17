"""Проба held() (turnir_train.py 964ecba, придержка (а)) в окружении юнита, как у turnir-e1: файл придержки — во временном
каталоге, юнит-пустышка probe-hold-dummy (sleep). Печатает 4 случая: активный юнит / неактивный / другая загрузка / нет файла."""
import pathlib, sys
sys.path.insert(0, sys.argv[1] + "/scripts")
import turnir_train as t
t.HOLD = pathlib.Path(sys.argv[2])
boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
res = {}
t.HOLD.write_text(f"1 {boot} probe-hold-dummy\n"); res["юнит активен"] = t.held()
t.HOLD.write_text(f"1 {boot} probe-hold-net-takogo\n"); res["юнита нет"] = t.held()
t.HOLD.write_text(f"1 00000000-0000-0000-0000-000000000000 probe-hold-dummy\n"); res["другая загрузка"] = t.held()
t.HOLD.unlink(); res["нет файла"] = t.held()
print(res)
assert res == {"юнит активен": True, "юнита нет": False, "другая загрузка": False, "нет файла": False}, res
print("held(): 4 случая из 4 — как в коде")
