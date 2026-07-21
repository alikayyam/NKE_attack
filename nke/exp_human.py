"""Human psychometric track driver (Sec 3.3). Computes the two labeled
recognizability proxies across eps_l for both NKE and Gaussian-control
conditions, and exports the stimulus set + crowdsourcing harness."""
import json, os
import numpy as np
import torch
from . import config as C
from . import models as M
from . import human as H
from .attackset import build_stimset


def run(dataset, arch, variant="nmifgm", n=200):
    ss = build_stimset(dataset, arch, variant=variant, n=n)
    x, y = ss["x"], ss["y"]

    # reference recognizer = the *other* architecture, trained independently,
    # never exposed to the attack gradients (generic-recognizer proxy).
    other = [a for a in M.ARCHS[dataset] if a != arch][0]
    ref = M.load(dataset, other, "clean")
    edge_net = H.train_edge_recognizer(dataset)

    out = {"dataset": dataset, "arch": arch, "ref_arch": other, "variant": variant,
           "levels": []}
    for eps in ss["grid"]:
        xa = ss["levels"][eps]["nke"]; xg = ss["levels"][eps]["gauss"]
        row = {
            "eps_l": eps,
            "model_acc": ss["levels"][eps]["model_acc"],
            "achieved_L2": float(ss["levels"][eps]["dist"].mean()),
            # proxy 1: generic independent recognizer
            "generic_proxy_nke": H.generic_recognizer_acc(ref, xa, y),
            "generic_proxy_gauss": H.generic_recognizer_acc(ref, xg, y),
            # proxy 2: shape / edge-map recognizer
            "shape_proxy_nke": H.shape_proxy_acc(edge_net, xa, y, dataset),
            "shape_proxy_gauss": H.shape_proxy_acc(edge_net, xg, y, dataset),
        }
        out["levels"].append(row)
        print(f"  eps_l={eps:5.1f} model_acc={row['model_acc']:.3f} | "
              f"generic(nke)={row['generic_proxy_nke']:.3f} generic(gauss)={row['generic_proxy_gauss']:.3f} | "
              f"shape(nke)={row['shape_proxy_nke']:.3f} shape(gauss)={row['shape_proxy_gauss']:.3f}")

    # export stimuli + harness for a real crowd run
    stim_dir = os.path.join(C.STIM_DIR, f"{dataset}_{arch}")
    H.export_stimuli_and_harness(ss, stim_dir, per_level=8)
    out["stimuli_dir"] = stim_dir

    path = os.path.join(C.RESULTS_DIR, f"human_{dataset}_{arch}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    for ds in ("mnist", "cifar10"):
        run(ds, list(M.ARCHS[ds].keys())[0])
