"""turnir_their.resolve_init: раннер передаёт --init каталогом предобучения (26.11: 55/55 IsADirectoryError в torch.load)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import turnir_their as tt  # noqa: E402


def test_resolve_init(tmp_path):
    pre = tmp_path / "seed0"
    (pre / "save_models").mkdir(parents=True)
    w = pre / "save_models" / "sam_slr_joint_all-7.pt"
    w.write_bytes(b"x")
    (pre / "summary.json").write_text(json.dumps({"weights": "save_models/sam_slr_joint_all-7.pt"}))
    assert tt.resolve_init(str(pre)) == str(w)  # каталог → выбранные веса
    assert tt.resolve_init(str(w)) == str(w)  # файл — как есть
    w.unlink()
    with pytest.raises(AssertionError):
        tt.resolve_init(str(pre))  # summary.json указывает на пропавший файл
