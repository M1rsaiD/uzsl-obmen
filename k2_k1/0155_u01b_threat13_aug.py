"""Дополнение к u01b_threat13.py (часть 3) — с аугментацией обучения: все записи ALL-UZ, один проход AugSet (затравка 48, как отбор
этапа 2; позиция = номер записи). Доля значений U01b на ±10 и доля кистей без ℓ_h (42 нуля по всем кадрам). Справочно, ЦП."""
import json, sys
from pathlib import Path
import numpy as np, torch
W = Path("/tmp/claude-1000/-home-error-404-uzsign/f3b24b76-183b-4505-805b-25c9b1ca2c12/scratchpad/wt_u01b")
OM = Path("/home/error_404/uzsign/osnovnaya_model")
sys.path[:0] = [str(W / "src"), str(W / "scripts"), str(OM.parent / "uzsign_e1/src"), str(OM.parent / "uzsign_ai_v2/src")]
import m13_train as T
from mpk import sohn
assert "u01b" in sohn.INPUT_TOKENS and Path(sohn.__file__).resolve().is_relative_to(W.resolve())
torch.set_num_threads(4)
TOK, NB = ("u01b",), 708
recs, gl, clips, _ = T.records("alluz", None, inp=True)
names = sorted(set(gl)); lab = {g: k for k, g in enumerate(names)}
recs.tokens = TOK
ds = T.AugSet(recs, np.array([lab[g] for g in gl]), 48, True)
n_clip = n_nz = n_all = 0; none = [0, 0]
for i in range(len(recs)):
    a, _ = ds[(i, i)]
    Fu = np.asarray(a)[:, NB:]
    assert Fu.shape[1] == 84
    n_clip += int((np.abs(Fu) >= sohn.EXTRA_CLIP).sum()); n_nz += int((Fu != 0).sum()); n_all += Fu.size
    none[0] += int(not Fu[:, :42].any()); none[1] += int(not Fu[:, 42:].any())
N = len(recs)
print(json.dumps({"записей": N, "с аугментацией (AugSet, затравка 48)": True,
                  "на ±10 / ненулевых значений U01b / доля": [n_clip, n_nz, n_clip / n_nz],
                  "на ±10 / всех значений / доля": [n_clip, n_all, n_clip / n_all],
                  "кистей без ℓ_h (42 нуля), столбцы Л / П / доля от 2·N": [none[0], none[1], (none[0] + none[1]) / (2 * N)]},
                 ensure_ascii=False, indent=1))
