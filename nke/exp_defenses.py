"""Defense-intervention track (Sec 3.5). Re-run the NKE attack white-box against
each defended model across eps_l; measure label-kept (success) rate. Also run a
correlation between each model's classical small-eps robust accuracy and its
resistance to the large-eps_l NKE attack."""
import json, os
import numpy as np
import torch
from . import config as C
from . import data as D
from . import models as M
from . import attack as A
from . import defenses as DEF
from .attackset import EPS_GRID


@torch.no_grad()
def _label_kept(model, x_adv, y, batch=128):
    preds = []
    for i in range(0, len(x_adv), batch):
        preds.append(model(x_adv[i:i + batch].to(C.DEVICE)).argmax(1).cpu())
    return float((torch.cat(preds) == y).float().mean())


@torch.no_grad()
def _robust_acc_classical(model, x, y, eps, batch=128):
    # measured OUTSIDE no_grad below
    pass


def robust_acc_classical(model, x, y, eps, steps=20):
    xadv = A.classical_pgd(model, x, y, eps=eps, steps=steps).cpu()
    return _label_kept(model, xadv, y)  # here label-kept == still-correct-under-attack = robust acc


def run(dataset, base_arch=None, variant="nmifgm", n=150, steps=40):
    if base_arch is None:
        base_arch = list(M.ARCHS[dataset].keys())[0]
    base = M.load(dataset, base_arch, "clean")
    x, y = D.sample_correct(base, dataset, n)
    grid = EPS_GRID[dataset]
    eps_c = 8 / 255 if dataset == "cifar10" else 0.3

    # candidate models/defenses
    defended = {"undefended": base}
    for dname in ["jpeg", "blur", "resize_pad", "smoothing"]:
        defended[dname] = DEF.build_defense(dname, base, dataset)
    adv_path = os.path.join(C.CKPT_DIR, f"{dataset}_{base_arch}_advtrain.pt")
    if os.path.exists(adv_path):
        defended["adv_trained"] = M.load(dataset, base_arch, "advtrain")

    out = {"dataset": dataset, "base_arch": base_arch, "variant": variant,
           "grid": grid, "defenses": {}, "correlation": {}}

    for name, mdl in defended.items():
        succ = []
        for eps in grid:
            if eps == 0:
                succ.append({"eps_l": 0.0, "success": 1.0}); continue
            # white-box against the (possibly defended) model
            xa = A.run_variant(mdl, x, y, eps, variant, steps=steps, seed=C.SEED).cpu()
            succ.append({"eps_l": eps, "success": _label_kept(mdl, xa, y)})
        # resistance, two operationalizations:
        #  (a) discrete: smallest eps_l where success drops below 0.5 (else max grid)
        #  (b) continuous: 1 - mean success over eps_l>0  (always has variance)
        resist = grid[-1]
        for s in succ:
            if s["success"] < 0.5 and s["eps_l"] > 0:
                resist = s["eps_l"]; break
        nz = [s["success"] for s in succ if s["eps_l"] > 0]
        resist_cont = float(1.0 - np.mean(nz))
        rob = robust_acc_classical(mdl, x, y, eps_c)
        out["defenses"][name] = {"success_curve": succ, "nke_resistance_eps": resist,
                                 "nke_resistance_cont": resist_cont,
                                 "classical_robust_acc": rob,
                                 "clean_acc": _label_kept(mdl, x, y)}
        print(f"  {name:12s} clean={out['defenses'][name]['clean_acc']:.3f} "
              f"classical_robust={rob:.3f} nke_resist_cont={resist_cont:.3f} "
              f"succ@max={succ[-1]['success']:.3f}")

    # correlation across models: classical robust acc vs continuous NKE resistance
    xs = [v["classical_robust_acc"] for v in out["defenses"].values()]
    ys = [v["nke_resistance_cont"] for v in out["defenses"].values()]
    ys_disc = [v["nke_resistance_eps"] for v in out["defenses"].values()]
    r = float(np.corrcoef(xs, ys)[0, 1]) if len(set(xs)) > 1 and len(set(ys)) > 1 else float("nan")
    out["correlation"] = {"pearson_r": r,
                          "classical_robust_acc": xs, "nke_resistance_cont": ys,
                          "nke_resistance_eps": ys_disc,
                          "models": list(out["defenses"].keys())}
    print(f"  correlation(classical_robust, nke_resistance_cont) r={r:.3f}")

    path = os.path.join(C.RESULTS_DIR, f"defenses_{dataset}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    for ds in ("mnist", "cifar10"):
        run(ds)
