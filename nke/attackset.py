"""Generate and cache a single shared set of NKE stimuli across an eps_l grid,
so the human, OOD, mechanism and defense tracks all operate on identical images
(the single-pipeline design of the paper's Fig. 1)."""
import os
import numpy as np
import torch
from . import config as C
from . import data as D
from . import models as M
from . import attack as A


# eps_l grids in L2 pixel-space (per-image L2 over all pixels in [0,1]).
# MNIST images are 1x28x28 (784 px); CIFAR 3x32x32 (3072 px). Grids are scaled
# so the largest value corresponds to heavy, clearly-visible degradation.
EPS_GRID = {
    "mnist":   [0.0, 1.0, 2.0, 3.5, 5.0, 7.0, 9.0],
    "cifar10": [0.0, 1.0, 2.5, 5.0, 8.0, 12.0, 16.0],
}


def cache_path(dataset, arch, variant, n):
    return os.path.join(C.RESULTS_DIR, f"stimset_{dataset}_{arch}_{variant}_n{n}.pt")


def build_stimset(dataset, arch, variant="nmifgm", n=200, steps=60, force=False):
    """Returns dict with clean x/y and, per eps_l: nke images, gaussian-control
    images, achieved distances, model preds/conf. Cached to disk."""
    path = cache_path(dataset, arch, variant, n)
    if os.path.exists(path) and not force:
        return torch.load(path)

    model = M.load(dataset, arch, "clean")
    x, y = D.sample_correct(model, dataset, n)
    grid = EPS_GRID[dataset]
    out = {"dataset": dataset, "arch": arch, "variant": variant,
           "x": x, "y": y, "grid": grid, "levels": {}}

    for eps in grid:
        if eps == 0.0:
            xa = x.clone()
        else:
            xa = A.run_variant(model, x, y, eps, variant, steps=steps, seed=C.SEED).cpu()
        # matched-budget Gaussian control at the same achieved L2 distance
        g = torch.Generator().manual_seed(C.SEED + int(eps * 100))
        noise = torch.randn(x.shape, generator=g)
        if eps == 0.0:
            xg = x.clone()
        else:
            nn_ = noise.flatten(1).norm(dim=1).clamp_min(1e-9)
            ach = (xa - x).flatten(1).norm(dim=1)  # match NKE achieved dist per image
            xg = (x + noise * (ach / nn_).view(-1, 1, 1, 1)).clamp(0, 1)

        with torch.no_grad():
            logits = model(xa.to(C.DEVICE))
            p = torch.softmax(logits, 1)
            pred = logits.argmax(1).cpu()
            conf = p.max(1).values.cpu()
        dist = (xa - x).flatten(1).norm(dim=1)
        out["levels"][eps] = {
            "nke": xa, "gauss": xg, "dist": dist,
            "pred": pred, "conf": conf,
            "model_acc": float((pred == y).float().mean()),
        }
        print(f"  [{dataset}/{arch}/{variant}] eps_l={eps:5.1f} "
              f"achieved_L2={dist.mean():5.2f} model_acc={out['levels'][eps]['model_acc']:.3f} "
              f"conf={conf.mean():.3f}")
    torch.save(out, path)
    print(f"[saved] {path}")
    return out
