"""Baseline replication (Sec 3.2): reduced cross-model white/black-box success
matrix + ablation sweeps over perturbation size, iterations, momentum mu."""
import json, os
import numpy as np
import torch
from . import config as C
from . import data as D
from . import models as M
from . import attack as A


@torch.no_grad()
def _label_kept_rate(model, x_adv, y):
    """Fraction of images the model still labels y (NKE success criterion)."""
    preds = []
    for i in range(0, len(x_adv), 256):
        preds.append(model(x_adv[i:i + 256].to(C.DEVICE)).argmax(1).cpu())
    preds = torch.cat(preds)
    return float((preds == y).float().mean())


def cross_model_matrix(dataset, archs, variant="nmifgm", n=150, eps_l=None, steps=60):
    """attack each source model; measure label-kept rate on every target model."""
    if eps_l is None:
        from .attackset import EPS_GRID
        eps_l = EPS_GRID[dataset][-2]  # a large but not maximal level
    mods = {a: M.load(dataset, a, "clean") for a in archs}
    matrix = {}
    for src in archs:
        x, y = D.sample_correct(mods[src], dataset, n)
        xadv = A.run_variant(mods[src], x, y, eps_l, variant, steps=steps, seed=C.SEED).cpu()
        matrix[src] = {tgt: _label_kept_rate(mods[tgt], xadv, y) for tgt in archs}
        print(f"  src={src} eps_l={eps_l}: " +
              " ".join(f"{t}={matrix[src][t]:.3f}{'*' if t==src else ''}" for t in archs))
    return {"dataset": dataset, "variant": variant, "eps_l": eps_l,
            "archs": archs, "matrix": matrix}


def ablation_sweeps(dataset, arch, variant="nmifgm", n=120):
    from .attackset import EPS_GRID
    model = M.load(dataset, arch, "clean")
    x, y = D.sample_correct(model, dataset, n)
    res = {"dataset": dataset, "arch": arch, "variant": variant}

    # 1) perturbation size
    res["eps_sweep"] = []
    for eps in EPS_GRID[dataset]:
        xa = x.clone() if eps == 0 else A.run_variant(model, x, y, eps, variant, steps=60, seed=C.SEED).cpu()
        res["eps_sweep"].append({"eps_l": eps, "success": _label_kept_rate(model, xa, y),
                                 "achieved_L2": float((xa - x).flatten(1).norm(dim=1).mean())})

    eps_fix = EPS_GRID[dataset][-2]
    # 2) iterations
    res["iter_sweep"] = []
    for steps in [5, 10, 20, 40, 80]:
        xa = A.run_variant(model, x, y, eps_fix, variant, steps=steps, seed=C.SEED).cpu()
        res["iter_sweep"].append({"steps": steps, "success": _label_kept_rate(model, xa, y)})

    # 3) momentum mu (only meaningful for momentum variants; sweep by overriding)
    res["mu_sweep"] = []
    for mu in [0.0, 0.3, 0.6, 0.9, 1.0]:
        xa = A.run_variant(model, x, y, eps_fix, variant, steps=60, momentum=mu, seed=C.SEED).cpu()
        res["mu_sweep"].append({"mu": mu, "success": _label_kept_rate(model, xa, y)})
    return res


def run(datasets=("mnist", "cifar10")):
    out = {}
    for ds in datasets:
        archs = list(M.ARCHS[ds].keys())
        print(f"[baseline] {ds} cross-model matrix")
        mat = cross_model_matrix(ds, archs)
        print(f"[baseline] {ds} ablation sweeps ({archs[0]})")
        abl = ablation_sweeps(ds, archs[0])
        out[ds] = {"matrix": mat, "ablation": abl}
    path = os.path.join(C.RESULTS_DIR, "baseline.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    run()
