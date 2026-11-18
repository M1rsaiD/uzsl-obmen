"""к3 0050: held() — дословно из хунка turnir_train.py диффа 0167 (bb3215c → b025809 = 964ecba), исполнение на ЦП с подставным
systemctl в PATH (код возврата задаётся переменной): активный юнит / неактивный / другая загрузка / нет файла / битый файл /
systemctl нет в PATH. Запуск: python3 -I held_k3.py <дифф 0167>"""
import os, re, subprocess, sys, tempfile, pathlib
d = open(sys.argv[1], encoding="utf-8").read()
part = next(p for p in re.split(r"(?m)^(?=diff --git )", d) if p.startswith("diff --git a/scripts/turnir_train.py "))
lines = part.splitlines(); i = next(k for k, l in enumerate(lines) if l.startswith("+def held"))
body = []
for l in lines[i:]:
    if not l.startswith("+"): break
    body.append(l[1:])
src = "\n".join(body).rstrip()
tmp = pathlib.Path(tempfile.mkdtemp()); stub = tmp / "bin"; stub.mkdir()
(stub / "systemctl").write_text('#!/bin/sh\nexit "${STUB_RC:-0}"\n'); (stub / "systemctl").chmod(0o755)
ns = {"Path": pathlib.Path, "subprocess": subprocess, "HOLD": tmp / "TURNIR_HOLD_E2"}
exec(src, ns); held = ns["held"]; H = ns["HOLD"]
boot = pathlib.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
os.environ["PATH"] = f"{stub}:{os.environ['PATH']}"
res = {}
H.write_text(f"1 {boot} turnir-e2\n"); os.environ["STUB_RC"] = "0"; res["юнит активен"] = held()
os.environ["STUB_RC"] = "3"; res["юнит неактивен"] = held()
os.environ["STUB_RC"] = "0"; H.write_text(f"1 00000000-0000-0000-0000-000000000000 turnir-e2\n"); res["другая загрузка"] = held()
H.write_text("мусор\n"); res["битый файл"] = held()
H.unlink(); res["нет файла"] = held()
print(res)
assert res == {"юнит активен": True, "юнит неактивен": False, "другая загрузка": False, "битый файл": False, "нет файла": False}
H.write_text(f"1 {boot} turnir-e2\n"); os.environ["PATH"] = "/nonexistent"
try:
    held(); print("systemctl нет в PATH: held() вернула значение")
except FileNotFoundError as e:
    print(f"systemctl нет в PATH: исключение {type(e).__name__} уходит из held() → упал бы run_job (остаток, разд. 1)")
