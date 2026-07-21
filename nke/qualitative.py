"""Qualitative example figures: clean -> NKE strips across eps_l, showing what
these large-perturbation, label-preserving images actually look like."""
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C
from . import models as M
from . import attack as A


def _show(ax, img):
    a = img.detach().cpu().numpy()
    if a.shape[0] == 3:
        ax.imshow(np.clip(np.transpose(a, (1, 2, 0)), 0, 1))
    else:
        ax.imshow(a[0], cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])


def strip_from_stimset(dataset, arch, n_examples=4):
    ss = torch.load(os.path.join(C.RESULTS_DIR,
          f"stimset_{dataset}_{arch}_nmifgm_n{'200' if dataset=='cifar10' else '150'}.pt"))
    grid = ss["grid"]; y = ss["y"]; classes = C.CLASS_NAMES[dataset]
    g = np.random.RandomState(C.SEED + 1)
    idx = g.choice(len(y), n_examples, replace=False)
    fig, axes = plt.subplots(n_examples, len(grid),
                             figsize=(1.15 * len(grid), 1.25 * n_examples))
    for r, j in enumerate(idx):
        for c, eps in enumerate(grid):
            ax = axes[r, c]
            _show(ax, ss["levels"][eps]["nke"][j])
            if r == 0:
                ax.set_title(f"$\\epsilon_l$={eps:g}", fontsize=8)
            if c == 0:
                ax.set_ylabel(classes[int(y[j])], fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle(f"{dataset.upper()} / {arch}: clean ($\\epsilon_l$=0) $\\to$ NKE. "
                 f"Model label & ~1.0 confidence preserved across the whole row.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(C.FIG_DIR, f"qualitative_{dataset}_{arch}.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}")


def strip_imagenet(n_examples=4):
    from . import imagenet_ext as IE
    models = IE.load_models(); src = models["resnet152"]
    x_all, y_all = IE.load_images(n_per_class=8)
    x, y = IE.correctly_classified(src, x_all, y_all)
    g = np.random.RandomState(C.SEED + 2)
    idx = g.choice(len(x), n_examples, replace=False)
    xs, ys = x[idx], y[idx]
    grid = IE.EPS_GRID_IN
    inv = {v: k for k, v in IE.WNID_TO_IDX.items()}
    names = {0: "tench", 217: "springer", 482: "cassette", 491: "chainsaw", 497: "church",
             566: "horn", 569: "garbage truck", 571: "gas pump", 574: "golf ball", 701: "parachute"}
    fig, axes = plt.subplots(n_examples, len(grid), figsize=(1.3 * len(grid), 1.4 * n_examples))
    for r in range(n_examples):
        xi = xs[r:r + 1]; yi = ys[r:r + 1]
        for c, eps in enumerate(grid):
            xa = xi.clone() if eps == 0 else A.run_variant(src, xi, yi, eps, "nmifgm", steps=40, seed=C.SEED).cpu()
            ax = axes[r, c]; _show(ax, xa[0])
            if r == 0:
                ax.set_title(f"$\\epsilon_l$={eps:g}", fontsize=8)
            if c == 0:
                ax.set_ylabel(names.get(int(yi), str(int(yi))), fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("ImageNet (ResNet-152): clean $\\to$ NKE. Model keeps its label at ~1.0 confidence throughout.",
                 fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p = os.path.join(C.FIG_DIR, "qualitative_imagenet.png")
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"[fig] {p}")


if __name__ == "__main__":
    strip_from_stimset("mnist", "lenet")
    strip_from_stimset("cifar10", "resnet18")
    strip_imagenet()
