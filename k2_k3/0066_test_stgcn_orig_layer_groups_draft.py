"""Черновик теста (поправка 4, П24): не в дереве OM; запуск из scratchpad с PYTHONPATH=OM/src:OM/tests."""
import torch

from test_baselines import CFGS, K, build  # OM/tests (pytest rootdir там же)


def test_stgcn_orig_layer_groups():
    """47ee784: 5 групп ST-GCN — блоки (10, 9) … (2, 1) с их edge_importance, data_bn — в нижней; edge_importance —
    nn.ParameterList (к1 0136 разд. 3); их fcn — Identity, голова — heads/emb/protos/part_heads _Base (группа 0, полный lr)."""
    from mpk.train import param_groups
    m = build(CFGS["stgcn_orig"], {"slovo": K, "pilot": 29})
    n = m.net
    assert isinstance(n.edge_importance, torch.nn.ParameterList) and len(n.edge_importance) == len(n.st_gcn_networks) == 10
    assert isinstance(n.fcn, torch.nn.Identity) and not list(n.fcn.parameters())
    g, groups = param_groups(m, {"backbone_lr_mult": 0.1, "freeze_steps": 5}, 1e-3)  # покрытие — assert внутри
    assert len(groups) == 5 and len(g) == 6
    for k, grp in enumerate(groups):
        i = 9 - 2 * k
        want = [n.st_gcn_networks[i], n.st_gcn_networks[i - 1], n.edge_importance[i], n.edge_importance[i - 1]]
        assert all(a is b for a, b in zip(grp[:4], want)) and len(grp) == (5 if k == 4 else 4)
    assert groups[-1][-1] is n.data_bn
    head = {id(p) for p in g[0]["params"]}
    assert head == {id(p) for name, p in m.named_parameters() if name.split(".")[0] in ("heads", "emb", "protos", "part_heads")}
    assert all(gr["lr"] == 1e-3 * 0.1 for gr in g[1:]) and g[0]["lr"] == 1e-3
