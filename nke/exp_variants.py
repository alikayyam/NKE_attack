"""Extension #1 (based on Nie et al.): compare the four attack variants
(I-FGSM, NI-FGM, NMI-FGSM, NMI-FGM) not on success rate alone (as Nie did) but
on our NEW metrics: OOD detectability, the human-proxy gap, and edge/texture.

To compare variants at *equal perturbation magnitude*, all four are projected to
the same L2 budget; they then differ only in step rule (sign vs L2-normalized
gradient) and momentum -- isolating exactly the factors Nie varied."""
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

# (label, sign, momentum) -- all L2-projected (p=2) for matched magnitude
VARIANTS = [
    ("I-FGSM",   True,  0.0),
    ("NI-FGM",   False, 0.0),
    ("NMI-FGSM", True,  1.0),
    ("NMI-FGM",  False, 1.0),
]
EPS = {"mnist": 5.0, "cifar10": 8.0}


def _edge_tex(x, xa):
    ef, tx = [], []
    for i in range(len(x)):
        c = x[i].numpy(); p = xa[i].numpy()
        c = np.transpose(c, (1, 2, 0)) if c.shape[0] == 3 else c[0]
        p = np.transpose(p, (1, 2, 0)) if p.shape[0] == 3 else p[0]
        f1, _ = MECH.edge_preservation(c, p); ef.append(f1)
        tx.append(MECH.texture_similarity(c, p))
    return float(np.mean(ef)), float(np.mean(tx))


def run(dataset, arch, n=200):
    model = M.load(dataset, arch, "clean")
    other = [a for a in M.ARCHS[dataset] if a != arch][0]
    ref = M.load(dataset, other, "clean")
    x, y = D.sample_correct(model, dataset, n)

    # OOD detectors fit on clean val
    xval, yval = D.sample_correct(model, dataset, 2000, seed=C.SEED + 7)
    maha = MET.Mahalanobis(model).fit(xval, yval.numpy())
    id_msp = MET.msp_score(model, xval); id_maha = maha.score(xval)
    thr_msp = MET.threshold_at_fpr(id_msp, 0.05); thr_maha = MET.threshold_at_fpr(id_maha, 0.05)

    eps = EPS[dataset]
    out = {"dataset": dataset, "arch": arch, "ref_arch": other, "eps_l": eps, "variants": []}
    for label, sign, mom in VARIANTS:
        xa = A.nke_attack(model, x, y, eps, steps=60, p=2, sign=sign,
                          momentum=mom, seed=C.SEED, sphere=True).cpu()
        succ = float((model(xa.to(C.DEVICE)).argmax(1).cpu() == y).float().mean())
        maha_det = MET.detection_rate(maha.score(xa), thr_maha)
        msp_det = MET.detection_rate(MET.msp_score(model, xa), thr_msp)
        gen = H.generic_recognizer_acc(ref, xa, y)
        ef, tx = _edge_tex(x, xa)
        row = {"variant": label, "sign": sign, "momentum": mom,
               "white_box_success": succ, "achieved_L2": float((xa - x).flatten(1).norm(dim=1).mean()),
               "maha_detect": maha_det, "msp_detect": msp_det,
               "generic_proxy": gen, "edge_f1": ef, "texture_glcm": tx}
        out["variants"].append(row)
        print(f"  {label:9s} succ={succ:.3f} L2={row['achieved_L2']:.2f} "
              f"maha_det={maha_det:.2f} msp_det={msp_det:.2f} "
              f"human_proxy={gen:.3f} edge={ef:.3f} tex={tx:.3f}")
    path = os.path.join(C.RESULTS_DIR, f"variants_{dataset}_{arch}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    for ds, arch in [("cifar10", "resnet18"), ("mnist", "lenet")]:
        print(f"=== {ds}/{arch} ===")
        run(ds, arch)
