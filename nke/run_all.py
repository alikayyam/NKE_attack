"""Master driver: runs all evaluation tracks on the trained models and builds
figures + a consolidated summary. Assumes clean (and optionally advtrain)
checkpoints already exist.

The ImageNet-scale and CLIP-proxy extensions require network downloads of
pretrained backbones (ResNet-152/VGG-16) and CLIP (LAION) weights, so they are
gated behind --heavy and skipped by default."""
import json, os, argparse, traceback
from . import config as C
from . import models as M
from . import (exp_baseline, exp_ood, exp_mechanism, exp_human, exp_defenses,
               exp_variants, exp_adaptive, exp_transfer_mech, exp_vit_shapebias,
               qualitative, figures)

PRIMARY = {"mnist": "lenet", "cifar10": "resnet18"}


def safe(fn, *a, **k):
    try:
        return fn(*a, **k)
    except Exception as e:
        print(f"[ERROR] {fn.__module__}.{fn.__name__}: {e}")
        traceback.print_exc()
        return None


def main(datasets, n, heavy=False):
    summary = {}
    # baseline (both datasets, all archs)
    summary["baseline"] = safe(exp_baseline.run, tuple(datasets))
    for ds in datasets:
        arch = PRIMARY[ds]
        print(f"\n===== {ds}/{arch}: OOD =====");          safe(exp_ood.run, ds, arch, n=n)
        print(f"\n===== {ds}/{arch}: mechanism =====");     safe(exp_mechanism.run, ds, arch, n=n)
        print(f"\n===== {ds}/{arch}: human =====");         safe(exp_human.run, ds, arch, n=n)
        print(f"\n===== {ds}/{arch}: variants =====");      safe(exp_variants.run, ds, arch, n=n)
        print(f"\n===== {ds}/{arch}: adaptive =====");      safe(exp_adaptive.run, ds, arch, n=n)
        print(f"\n===== {ds}: defenses =====");             safe(exp_defenses.run, ds)
        safe(qualitative.strip_from_stimset, ds, arch)
    # transfer-mechanism analysis (CIFAR-10 only: needs a second architecture)
    if "cifar10" in datasets:
        print("\n===== cifar10/resnet18: transfer-mechanism =====")
        safe(exp_transfer_mech.run, "cifar10", "resnet18")
        print("\n===== cifar10/resnet18: adaptive vs. detector battery =====")
        safe(exp_adaptive.run_strong_detectors, "cifar10", "resnet18", n)
        print("\n===== cifar10/resnet18 (adv-trained): adaptive =====")
        safe(exp_adaptive.run, "cifar10", "resnet18", n, "advtrain")
        # ViT shape-bias probe: run the standard tracks with vit as source, plus
        # the cross-recognizer contrast (requires a trained cifar10_vit_clean.pt).
        if "vit" in M.ARCHS["cifar10"]:
            for track, fn in (("OOD", exp_ood.run), ("mechanism", exp_mechanism.run),
                              ("human", exp_human.run), ("adaptive", exp_adaptive.run)):
                print(f"\n===== cifar10/vit: {track} =====")
                safe(fn, "cifar10", "vit", n=n) if track != "adaptive" else safe(fn, "cifar10", "vit", n)
            print("\n===== cifar10: ViT cross-recognizer shape-bias probe =====")
            safe(exp_vit_shapebias.run, "cifar10", "nmifgm", n)
    if heavy:
        print("\n===== ImageNet-scale + CLIP proxy (downloads pretrained weights) =====")
        from . import imagenet_ext, clip_proxy
        safe(imagenet_ext.run)
        safe(clip_proxy.run_cifar)
        safe(clip_proxy.run_imagenet)
        safe(qualitative.strip_imagenet)
    figures.make_all()
    print("\n[run_all] done.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10"])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--heavy", action="store_true",
                    help="also run ImageNet-scale + CLIP extensions (network downloads)")
    a = ap.parse_args()
    main(a.datasets, a.n, heavy=a.heavy)
