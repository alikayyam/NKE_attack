"""Produce the overlay figures and summary tables from saved results JSON."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3})


def _load(name):
    p = os.path.join(C.RESULTS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def overlay_figure(dataset, arch):
    """The paper's Fig. 4: model success, human proxies, OOD flag rate on one axis."""
    ood = _load(f"ood_{dataset}_{arch}.json")
    hum = _load(f"human_{dataset}_{arch}.json")
    mech = _load(f"mechanism_{dataset}_{arch}.json")
    if not (ood and hum):
        print(f"[fig] missing results for {dataset}/{arch}"); return
    eps = [r["eps_l"] for r in ood["levels"]]
    macc = [r["model_acc"] for r in ood["levels"]]
    msp = [r["msp_detect"] for r in ood["levels"]]
    maha = [r["maha_detect"] for r in ood["levels"]]
    energy = [r["energy_detect"] for r in ood["levels"]]
    gen = [r["generic_proxy_nke"] for r in hum["levels"]]
    shape = [r["shape_proxy_nke"] for r in hum["levels"]]
    gen_g = [r["generic_proxy_gauss"] for r in hum["levels"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(eps, macc, "-o", color="tab:blue", lw=2.2, label="Model accuracy (label kept)")
    ax.plot(eps, gen, "--s", color="tab:red", label="Human proxy: generic recognizer (NKE)")
    ax.plot(eps, shape, "--^", color="darkred", label="Human proxy: shape/edge recognizer (NKE)")
    ax.plot(eps, gen_g, ":x", color="gray", label="Generic recognizer (Gaussian control)")
    ax.plot(eps, maha, "-.D", color="tab:green", label="OOD flag rate: Mahalanobis")
    ax.plot(eps, msp, "-.v", color="tab:olive", label="OOD flag rate: MSP")
    ax.plot(eps, energy, "-.P", color="tab:cyan", label="OOD flag rate: Energy")
    ax.set_xlabel(r"Perturbation magnitude $\epsilon_l$ (L2)")
    ax.set_ylabel("Rate / Accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"NKE overlay — {dataset.upper()} / {arch}")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"overlay_{dataset}_{arch}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p}")


def mechanism_figure(dataset, arch):
    mech = _load(f"mechanism_{dataset}_{arch}.json")
    ood = _load(f"ood_{dataset}_{arch}.json")
    if not mech:
        return
    eps = [r["eps_l"] for r in mech["levels"]]
    edge = [r["edge_preservation_f1"] for r in mech["levels"]]
    tex = [r["texture_similarity_glcm"] for r in mech["levels"]]
    gram = [r["texture_similarity_gram"] for r in mech["levels"]]
    macc = [r["model_acc"] for r in mech["levels"]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(eps, macc, "-o", color="tab:blue", lw=2, label="Model accuracy")
    ax.plot(eps, edge, "--^", color="darkgreen", label="Edge preservation (Canny F1)")
    ax.plot(eps, tex, "-.s", color="tab:orange", label="Texture similarity (GLCM)")
    ax.plot(eps, gram, ":d", color="chocolate", label="Texture similarity (Gram)")
    ax.set_xlabel(r"Perturbation magnitude $\epsilon_l$ (L2)")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Shape vs texture — {dataset.upper()} / {arch}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"mechanism_{dataset}_{arch}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def calibration_figure(dataset, arch):
    ood = _load(f"ood_{dataset}_{arch}.json")
    if not ood:
        return
    eps = [r["eps_l"] for r in ood["levels"]]
    conf = [r["mean_conf"] for r in ood["levels"]]
    acc = [r["model_acc"] for r in ood["levels"]]
    ece = [r["ece"] for r in ood["levels"]]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(eps, acc, "-o", label="Accuracy", color="tab:blue")
    ax.plot(eps, conf, "-s", label="Mean confidence", color="tab:red")
    ax.plot(eps, ece, "--^", label="ECE", color="tab:purple")
    ax.set_xlabel(r"$\epsilon_l$ (L2)"); ax.set_ylabel("Value"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Confidence vs accuracy — {dataset.upper()} / {arch}")
    ax.legend(fontsize=8); fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"calibration_{dataset}_{arch}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def defense_figure(dataset):
    d = _load(f"defenses_{dataset}.json")
    if not d:
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, v in d["defenses"].items():
        eps = [s["eps_l"] for s in v["success_curve"]]
        su = [s["success"] for s in v["success_curve"]]
        a1.plot(eps, su, "-o", label=name, ms=4)
    a1.set_xlabel(r"$\epsilon_l$ (L2)"); a1.set_ylabel("NKE success (label kept)")
    a1.set_title(f"Attack success under defenses — {dataset.upper()}")
    a1.legend(fontsize=7); a1.set_ylim(-0.03, 1.05)
    cor = d["correlation"]
    yv = cor.get("nke_resistance_cont", cor.get("nke_resistance_eps"))
    a2.scatter(cor["classical_robust_acc"], yv, color="tab:red", zorder=3)
    a2.set_xlabel("Classical robust acc (small-$\\epsilon$ PGD)")
    a2.set_ylabel(r"NKE resistance (1 - mean success)")
    a2.set_title(f"Robustness correlation (r={cor['pearson_r']:.2f})")
    a2.margins(0.15)
    texts = [a2.annotate(m, (cor["classical_robust_acc"][i], yv[i]), fontsize=9,
                          bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
             for i, m in enumerate(cor["models"])]
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=a2, x=cor["classical_robust_acc"], y=yv,
                    arrowprops=dict(arrowstyle="-", color="gray", lw=0.6),
                    expand=(1.4, 1.6), force_text=(0.6, 0.8))
    except ImportError:
        pass
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"defenses_{dataset}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def variants_figure(dataset, arch):
    v = _load(f"variants_{dataset}_{arch}.json")
    if not v:
        return
    labels = [r["variant"] for r in v["variants"]]
    metrics = [("white_box_success", "Model success"), ("maha_detect", "Mahalanobis detect"),
               ("msp_detect", "MSP detect"), ("generic_proxy", "Human proxy (generic)"),
               ("edge_f1", "Edge preservation")]
    x = np.arange(len(labels)); w = 0.16
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for i, (k, lab) in enumerate(metrics):
        ax.bar(x + (i - 2) * w, [r[k] for r in v["variants"]], w, label=lab)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Rate / accuracy")
    ax.set_title(f"Attack variants at matched $L_2$ — {dataset.upper()} / {arch} "
                 f"($L_2\\approx{v['variants'][0]['achieved_L2']:.0f}$)")
    ax.legend(fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"variants_{dataset}_{arch}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def transfer_mech_figure(dataset, src):
    d = _load(f"transfer_mech_{dataset}_{src}.json")
    if not d:
        return
    eps = [r["eps_l"] for r in d["levels"]]
    auroc = [r["auroc_proximity_predicts_transfer"] for r in d["levels"]]
    tr = [r["transfer_rate"] for r in d["levels"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(eps, auroc, "-o", color="tab:purple", lw=2,
            label="AUROC: manifold-proximity → transfer")
    ax.axhline(0.5, ls=":", color="gray", label="chance (0.5)")
    ax.plot(eps, tr, "--s", color="tab:blue", label="black-box transfer rate")
    ax.set_xlabel(r"Perturbation magnitude $\epsilon_l$ (L2)")
    ax.set_ylabel("Value"); ax.set_ylim(0.3, 1.02)
    ax.set_title(f"Does proximity predict transfer? — {dataset.upper()} ({src}$\\to$other)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"transfer_mech_{dataset}_{src}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def imagenet_figure():
    d = _load("imagenet_ext.json")
    if not d:
        return
    eps = [r["eps_l"] for r in d["levels"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(eps, [r["white_box_success"] for r in d["levels"]], "-o", color="tab:blue", lw=2.2,
            label="White-box success (model)")
    ax.plot(eps, [r["transfer_proxy"] for r in d["levels"]], "--s", color="tab:red",
            label=f"Transfer to {d['tgt']} (human proxy, NKE)")
    if "transfer_proxy_gauss" in d["levels"][0]:
        ax.plot(eps, [r["transfer_proxy_gauss"] for r in d["levels"]], ":x", color="gray",
                label="Generic recognizer (Gaussian control)")
    ax.plot(eps, [r["maha_detect"] for r in d["levels"]], "-.D", color="tab:green",
            label="OOD: Mahalanobis")
    ax.plot(eps, [r["msp_detect"] for r in d["levels"]], "-.v", color="tab:olive", label="OOD: MSP")
    ax.plot(eps, [r["texture_glcm"] for r in d["levels"]], ":P", color="tab:orange",
            label="Texture similarity (GLCM)")
    ax.plot(eps, [r["edge_f1"] for r in d["levels"]], ":^", color="darkgreen",
            label="Edge preservation")
    ax.set_xlabel(r"Perturbation magnitude $\epsilon_l$ (L2)")
    ax.set_ylabel("Rate / score"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"ImageNet scale ({d['src']}$\\to${d['tgt']}, real Imagenette images)")
    ax.legend(fontsize=7, loc="center right")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, "imagenet_ext.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def complexity_trend_figure():
    """Black-box transfer vs task complexity, measured on all three scales."""
    import json
    b = _load("baseline.json"); ie = _load("imagenet_ext.json")
    pts = []
    if b:
        m = b["mnist"]["matrix"]["matrix"]; a = b["mnist"]["matrix"]["archs"]
        pts.append(("MNIST", np.mean([m[a[0]][a[1]], m[a[1]][a[0]]])))
        m = b["cifar10"]["matrix"]["matrix"]; a = b["cifar10"]["matrix"]["archs"]
        pts.append(("CIFAR-10", np.mean([m[a[0]][a[1]], m[a[1]][a[0]]])))
    if ie:
        # transfer at a large, clearly-visible perturbation (eps_l=30, L2~61)
        row = [r for r in ie["levels"] if r["eps_l"] == 30.0]
        pts.append(("ImageNet", row[0]["transfer_proxy"] if row else ie["levels"][-2]["transfer_proxy"]))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(range(len(pts)), [p[1] for p in pts], "-o", color="tab:purple", lw=2, ms=8)
    ax.set_xticks(range(len(pts))); ax.set_xticklabels([p[0] for p in pts])
    ax.set_ylabel("Black-box transfer (label kept)"); ax.set_ylim(-0.03, 1.02)
    ax.set_xlabel("Increasing task complexity $\\rightarrow$")
    ax.set_title("NKE transferability collapses with task complexity")
    for i, p in enumerate(pts):
        ax.annotate(f"{p[1]:.2f}", (i, p[1]), textcoords="offset points", xytext=(0, 8), fontsize=9)
    fig.tight_layout()
    pth = os.path.join(C.FIG_DIR, "complexity_trend.png")
    fig.savefig(pth); plt.close(fig)
    print(f"[fig] {pth}")


def clip_figure():
    cf = _load("clip_cifar10.json"); imf = _load("clip_imagenet.json")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, d, title in [(axes[0], cf, "CIFAR-10"), (axes[1], imf, "ImageNet")]:
        if not d:
            continue
        eps = [r["eps_l"] for r in d["levels"]]
        ax.plot(eps, [1.0 for _ in eps], "-o", color="tab:blue", lw=2, label="Model (source)")
        ax.plot(eps, [r["clip_nke"] for r in d["levels"]], "--s", color="tab:red",
                label="CLIP zero-shot (NKE)")
        ax.plot(eps, [r["clip_gauss"] for r in d["levels"]], ":x", color="gray",
                label="CLIP zero-shot (Gaussian ctrl)")
        ax.axhline(0.1, ls=":", color="0.7", lw=1)
        ax.set_title(title); ax.set_xlabel(r"$\epsilon_l$ (L2)"); ax.set_ylim(-0.03, 1.05)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Accuracy")
    fig.suptitle("CLIP as a stronger human-recognizability proxy (chance = 0.10)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(C.FIG_DIR, "clip_proxy.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def adaptive_figure(dataset, arch):
    """Adaptive attack vs. Mahalanobis: success and detection as lambda grows."""
    d = _load(f"adaptive_{dataset}_{arch}.json")
    if not d:
        return
    lam = [r["lambda"] for r in d["lambdas"]]
    xs = np.arange(len(lam))                        # log-ish spacing -> use index axis
    succ = [r["success"] for r in d["lambdas"]]
    mwb = [r["maha_detect"] for r in d["lambdas"]]
    mho = [r.get("maha_detect_heldout") for r in d["lambdas"]]
    msp = [r["msp_detect"] for r in d["lambdas"]]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, succ, "-o", color="tab:blue", lw=2.2, label="NKE success (label kept)")
    ax.plot(xs, mwb, "-.D", color="tab:green", label="Mahalanobis detect (white-box)")
    if all(v is not None for v in mho):
        ax.plot(xs, mho, "--s", color="seagreen", label="Mahalanobis detect (held-out)")
    ax.plot(xs, msp, "-.v", color="tab:olive", label="MSP detect")
    ax.axvspan(0.5, 3.5, color="gold", alpha=0.15)   # the fully-successful (100%) evasive band
    ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in lam], rotation=45)
    ax.set_xlabel(r"Adaptive penalty weight $\lambda$")
    ax.set_ylabel("Rate / accuracy"); ax.set_ylim(-0.03, 1.05)
    ax.set_title(f"Adaptive evasion of Mahalanobis — {dataset.upper()} / {arch} "
                 f"($L_2\\approx{d['lambdas'][0]['achieved_L2']:.0f}$)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"adaptive_{dataset}_{arch}.png")
    fig.savefig(p); plt.close(fig)
    print(f"[fig] {p}")


def vit_shapebias_figure(dataset="cifar10"):
    d = _load(f"vit_shapebias_{dataset}.json")
    if not d:
        return
    archs = d["archs"]; grid = d["grid"]; cr = d["cross_recognizer"]
    cnns = [a for a in archs if a != "vit"]
    base = {(s, r): cr[0]["R"][s][r] for s in archs for r in archs}   # eps=0

    def ret(eps, s, r):
        b = base[(s, r)]
        e = next(x for x in cr if x["eps_l"] == eps)["R"][s][r]
        return e / b if b > 0 else float("nan")

    colors = {"resnet18": "tab:blue", "vgg11": "tab:orange", "vit": "tab:green"}
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(15.5, 4.3))

    # Panel 1: retention of CNN-sourced NKE — ViT recognizer vs other-CNN recognizer.
    for src in cnns:
        othercnn = [c for c in cnns if c != src]
        vit_curve = [ret(g, src, "vit") for g in grid]
        cnn_curve = [float(np.mean([ret(g, src, o) for o in othercnn])) for g in grid]
        a1.plot(grid, vit_curve, "-o", color=colors[src], lw=2, label=f"{src}→ViT")
        a1.plot(grid, cnn_curve, "--s", color=colors[src], lw=1.6, alpha=0.7, label=f"{src}→CNN")
    a1.set_xlabel(r"$\epsilon_l$ (L2)"); a1.set_ylabel("Retention (acc / clean-acc)")
    a1.set_ylim(-0.03, 1.1)
    a1.set_title("Reads CNN-sourced NKE\n(retention-normalised)")
    a1.legend(fontsize=7.5)

    # Panel 2: the shape-bias contrast (ViT retention minus CNN retention).
    contrast = d["shape_bias_contrast"]
    ce = [r["eps_l"] for r in contrast]
    cv = [r["mean_vit_minus_cnn_retention"] for r in contrast]
    a2.axhline(0, color="gray", lw=1, ls=":")
    a2.plot(ce, cv, "-D", color="tab:purple", lw=2)
    a2.fill_between(ce, 0, cv, where=[v is not None and v < 0 for v in cv],
                    color="tab:red", alpha=0.13)
    a2.set_xlabel(r"$\epsilon_l$ (L2)"); a2.set_ylabel("ViT ret. − CNN ret.")
    a2.set_title("Shape-bias prediction test\n(>0 would confirm; observed <0)")

    # Panel 3: source specificity — how transferable is NKE *from* each source.
    for src in archs:
        curve = [next(r for r in d["source_specificity"] if r["eps_l"] == g)[src] for g in grid]
        a3.plot(grid, curve, "-o", color=colors[src], lw=2, label=f"source={src}")
    a3.set_xlabel(r"$\epsilon_l$ (L2)")
    a3.set_ylabel("Mean cross-model retention")
    a3.set_ylim(-0.03, 1.1)
    a3.set_title("Off-manifold specificity by source\n(lower = more model-specific)")
    a3.legend(fontsize=7.5)

    fig.suptitle(f"ViT shape-bias probe — {dataset.upper()}", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(C.FIG_DIR, f"vit_shapebias_{dataset}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[fig] {p}")


def make_all():
    vit_shapebias_figure("cifar10")
    # ViT-source track figures (early-return if the JSON isn't present yet)
    for fn in (overlay_figure, mechanism_figure, calibration_figure):
        try:
            fn("cifar10", "vit")
        except Exception as e:
            print(f"[fig] skip vit {fn.__name__}: {e}")
    adaptive_figure("cifar10", "resnet18")
    variants_figure("cifar10", "resnet18"); variants_figure("mnist", "lenet")
    transfer_mech_figure("cifar10", "resnet18")
    imagenet_figure(); complexity_trend_figure(); clip_figure()
    for ds in ("mnist", "cifar10"):
        arch = "lenet" if ds == "mnist" else "resnet18"
        overlay_figure(ds, arch); mechanism_figure(ds, arch)
        calibration_figure(ds, arch); defense_figure(ds)


if __name__ == "__main__":
    make_all()
