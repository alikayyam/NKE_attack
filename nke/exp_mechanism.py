"""Shape-vs-texture mechanistic experiment (Sec 3.4). For each eps_l, compute
mean edge-preservation (Canny F1) and texture-similarity (GLCM + Gram) between
clean and NKE-perturbed images, alongside model accuracy."""
import json, os
import numpy as np
from . import config as C
from . import models as M
from . import mechanism as MECH
from .attackset import build_stimset


def _img_np(t):
    a = t.numpy()
    return a  # CxHxW in [0,1]


def run(dataset, arch, variant="nmifgm", n=200):
    ss = build_stimset(dataset, arch, variant=variant, n=n)
    x = ss["x"]
    out = {"dataset": dataset, "arch": arch, "variant": variant, "levels": []}
    for eps in ss["grid"]:
        xa = ss["levels"][eps]["nke"]
        edge_f1, tex, gram = [], [], []
        for i in range(len(x)):
            c = _img_np(x[i]); p = _img_np(xa[i])
            f1, _ = MECH.edge_preservation(np.transpose(c, (1, 2, 0)) if c.shape[0] == 3 else c[0],
                                           np.transpose(p, (1, 2, 0)) if p.shape[0] == 3 else p[0])
            edge_f1.append(f1)
            tex.append(MECH.texture_similarity(np.transpose(c, (1, 2, 0)) if c.shape[0] == 3 else c[0],
                                               np.transpose(p, (1, 2, 0)) if p.shape[0] == 3 else p[0]))
            gram.append(MECH.gram_cosine(np.transpose(c, (1, 2, 0)) if c.shape[0] == 3 else c[0],
                                         np.transpose(p, (1, 2, 0)) if p.shape[0] == 3 else p[0]))
        row = {"eps_l": eps, "model_acc": ss["levels"][eps]["model_acc"],
               "edge_preservation_f1": float(np.mean(edge_f1)),
               "texture_similarity_glcm": float(np.mean(tex)),
               "texture_similarity_gram": float(np.mean(gram)),
               "achieved_L2": float(ss["levels"][eps]["dist"].mean())}
        out["levels"].append(row)
        print(f"  eps_l={eps:5.1f} acc={row['model_acc']:.3f} "
              f"edge_F1={row['edge_preservation_f1']:.3f} "
              f"tex_glcm={row['texture_similarity_glcm']:.3f} "
              f"tex_gram={row['texture_similarity_gram']:.3f}")
    path = os.path.join(C.RESULTS_DIR, f"mechanism_{dataset}_{arch}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    for ds in ("mnist", "cifar10"):
        run(ds, list(M.ARCHS[ds].keys())[0])
