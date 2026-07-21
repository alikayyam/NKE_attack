"""OOD detection + calibration track (Sec 3.3 of paper / our Sec ood).

For each eps_l level of the shared stimulus set, report:
  - detection rate of MSP / energy / Mahalanobis at a threshold fixed to 5% FPR
    on clean in-distribution validation data,
  - AUROC vs clean,
  - ECE of the model's (confident, mostly-correct) predictions.
Reference points: classical small-eps PGD adversarials and genuine OOD (SVHN/FMNIST).
"""
import json, os
import numpy as np
import torch
from . import config as C
from . import data as D
from . import models as M
from . import attack as A
from . import metrics as MET
from .attackset import build_stimset


def _confs_correct(model, x, y, batch=256):
    import torch.nn.functional as F
    confs, corr = [], []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(C.DEVICE)
            p = F.softmax(model(xb), 1)
            c = p.max(1)
            confs.append(c.values.cpu()); corr.append((c.indices.cpu() == y[i:i + batch]))
    return torch.cat(confs).numpy(), torch.cat(corr).numpy()


def run(dataset, arch, variant="nmifgm", n=200):
    model = M.load(dataset, arch, "clean")
    ss = build_stimset(dataset, arch, variant=variant, n=n)
    x, y = ss["x"], ss["y"]

    # --- fit detectors on one clean split, calibrate the 5%-FPR threshold on a
    #     DISJOINT clean split (an in-sample threshold on the fit set does not
    #     generalise: a fitted-covariance Mahalanobis overfits, so its in-sample
    #     5% threshold corresponds to a much higher real FPR). AUROC is reported
    #     against the same held-out clean reference and is threshold-independent.
    xval, yval = D.sample_correct(model, dataset, 2000, seed=C.SEED + 7)     # fit
    xcal, _ = D.sample_correct(model, dataset, 2000, seed=C.SEED + 123)      # calibrate (held-out)
    maha = MET.Mahalanobis(model).fit(xval, yval.numpy())
    id_scores = {
        "msp": MET.msp_score(model, xcal),
        "energy": MET.energy_score(model, xcal),
        "maha": maha.score(xcal),
    }
    thr = {k: MET.threshold_at_fpr(v, 0.05) for k, v in id_scores.items()}

    # --- genuine OOD reference ---
    ood_ds = D.get_ood_dataset(dataset)
    xo = torch.stack([ood_ds[i][0] for i in range(min(500, len(ood_ds)))])
    if dataset == "mnist" and xo.shape[1] == 1:
        pass
    ref_ood = {
        "msp": MET.msp_score(model, xo), "energy": MET.energy_score(model, xo),
        "maha": maha.score(xo),
    }

    # --- classical small-eps adversarial reference ---
    eps_c = 8 / 255 if dataset == "cifar10" else 0.3
    xadv_c = A.classical_pgd(model, x, y, eps=eps_c, steps=20).cpu()
    ref_adv = {
        "msp": MET.msp_score(model, xadv_c), "energy": MET.energy_score(model, xadv_c),
        "maha": maha.score(xadv_c),
    }
    cc, cor = _confs_correct(model, xadv_c, y)
    adv_ece, _ = MET.expected_calibration_error(cc, cor)

    out = {"dataset": dataset, "arch": arch, "variant": variant,
           "thresholds_fpr5": thr, "levels": [],
           "reference": {}}

    for name, sc in [("genuine_ood", ref_ood), ("classical_adv", ref_adv)]:
        out["reference"][name] = {
            det: {"detect_rate": MET.detection_rate(sc[det], thr[det]),
                  "auroc": MET.auroc(id_scores[det], sc[det])} for det in thr
        }
    out["reference"]["classical_adv"]["ece"] = adv_ece
    out["reference"]["classical_adv"]["model_acc"] = float(cor.mean())

    # --- per-eps_l NKE stimuli ---
    for eps in ss["grid"]:
        xa = ss["levels"][eps]["nke"]
        sc = {"msp": MET.msp_score(model, xa), "energy": MET.energy_score(model, xa),
              "maha": maha.score(xa)}
        confs, correct = _confs_correct(model, xa, y)
        ece, _ = MET.expected_calibration_error(confs, correct)
        row = {"eps_l": eps, "model_acc": ss["levels"][eps]["model_acc"],
               "mean_conf": float(confs.mean()), "ece": ece,
               "achieved_L2": float(ss["levels"][eps]["dist"].mean())}
        for det in thr:
            row[f"{det}_detect"] = MET.detection_rate(sc[det], thr[det])
            row[f"{det}_auroc"] = MET.auroc(id_scores[det], sc[det])
        out["levels"].append(row)
        print(f"  eps_l={eps:5.1f} acc={row['model_acc']:.3f} conf={row['mean_conf']:.3f} "
              f"ece={ece:.3f} msp_det={row['msp_detect']:.2f} maha_det={row['maha_detect']:.2f} "
              f"energy_det={row['energy_detect']:.2f}")

    path = os.path.join(C.RESULTS_DIR, f"ood_{dataset}_{arch}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    for ds in ("mnist", "cifar10"):
        run(ds, list(M.ARCHS[ds].keys())[0])
