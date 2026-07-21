"""Extension #2 (based on Nie et al.): Nie *reported* the white-box>>black-box
transfer collapse but did not explain it. Here we ask WHICH examples transfer.

Hypothesis: the rare NKE examples that transfer to a second architecture are the
ones that stayed closer to the natural manifold (lower Mahalanobis feature
distance) and retained more shape/edge structure -- i.e. transferability, OOD
distance, and human recognizability are three views of the same "on-manifold-ness".

We generate NKE on a source model, evaluate transfer to a target model per image,
and test whether the source-model Mahalanobis score and edge-preservation predict
transfer (group means, point-biserial r, and AUROC)."""
import json, os
import numpy as np
import torch
from . import config as C
from . import data as D
from . import models as M
from . import attack as A
from . import metrics as MET
from . import mechanism as MECH
from . import human as H


def _edge_scores(x, xa):
    out = []
    for i in range(len(x)):
        c = x[i].numpy(); p = xa[i].numpy()
        c = np.transpose(c, (1, 2, 0)) if c.shape[0] == 3 else c[0]
        p = np.transpose(p, (1, 2, 0)) if p.shape[0] == 3 else p[0]
        f1, _ = MECH.edge_preservation(c, p); out.append(f1)
    return np.array(out)


def _point_biserial(binary, cont):
    b = np.asarray(binary, float); c = np.asarray(cont, float)
    if len(np.unique(b)) < 2:
        return float("nan")
    return float(np.corrcoef(b, c)[0, 1])


def run(dataset, src_arch, n=400, eps_candidates=None):
    tgt_arch = [a for a in M.ARCHS[dataset] if a != src_arch][0]
    src = M.load(dataset, src_arch, "clean"); tgt = M.load(dataset, tgt_arch, "clean")
    ref = tgt  # generic recognizer = target model
    x, y = D.sample_correct(src, dataset, n)

    # Mahalanobis on the SOURCE model's features
    xval, yval = D.sample_correct(src, dataset, 2000, seed=C.SEED + 7)
    maha = MET.Mahalanobis(src).fit(xval, yval.numpy())

    from .attackset import EPS_GRID
    if eps_candidates is None:
        eps_candidates = EPS_GRID[dataset][2:]  # skip tiny levels
    out = {"dataset": dataset, "src": src_arch, "tgt": tgt_arch, "levels": []}
    for eps in eps_candidates:
        xa = A.run_variant(src, x, y, eps, "nmifgm", steps=60, seed=C.SEED).cpu()
        wb = (src(xa.to(C.DEVICE)).argmax(1).cpu() == y)          # white-box kept
        transfer = (tgt(xa.to(C.DEVICE)).argmax(1).cpu() == y).numpy().astype(int)  # transfers?
        maha_s = maha.score(xa)                                    # feature distance (higher=OOD)
        edge_s = _edge_scores(x, xa)                               # shape preserved (higher=more)
        tr_rate = float(transfer.mean())
        # only meaningful where transfer is partial
        row = {
            "eps_l": eps, "white_box_success": float(wb.float().mean()),
            "transfer_rate": tr_rate,
            "maha_transferred": float(maha_s[transfer == 1].mean()) if transfer.sum() else None,
            "maha_not_transferred": float(maha_s[transfer == 0].mean()) if (transfer == 0).sum() else None,
            "edge_transferred": float(edge_s[transfer == 1].mean()) if transfer.sum() else None,
            "edge_not_transferred": float(edge_s[transfer == 0].mean()) if (transfer == 0).sum() else None,
            # transfer(1) should correlate NEGATIVELY with maha (closer=transfers), POS with edge
            "r_transfer_vs_maha": _point_biserial(transfer, maha_s),
            "r_transfer_vs_edge": _point_biserial(transfer, edge_s),
            # AUROC: does proximity (-maha) rank transferred examples above non-transferred?
            "auroc_proximity_predicts_transfer": (
                __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(transfer, -maha_s)
                if (transfer.sum() and (transfer == 0).sum()) else float("nan")),
        }
        out["levels"].append(row)
        print(f"  eps_l={eps:5.1f} transfer={tr_rate:.2f} | "
              f"maha T={row['maha_transferred']} NT={row['maha_not_transferred']} | "
              f"edge T={row['edge_transferred']} NT={row['edge_not_transferred']} | "
              f"r(maha)={row['r_transfer_vs_maha']:.3f} r(edge)={row['r_transfer_vs_edge']:.3f}")
    path = os.path.join(C.RESULTS_DIR, f"transfer_mech_{dataset}_{src_arch}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    run("cifar10", "resnet18")
