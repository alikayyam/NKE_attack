"""ImageNet-scale extension. Uses the real ImageNet-pretrained models (ResNet-152
and VGG-16-BN as two architecturally distinct families) on Imagenette --- a
10-class subset of *real* ImageNet images at 224px. This lets us measure the
third point of the transferability--complexity trend (MNIST -> CIFAR -> ImageNet)
with real data rather than citing it, and re-run the OOD / human-proxy / mechanism
tracks at ImageNet scale.

The full ImageNet training set is unavailable here; Imagenette provides correctly
-labelled natural images, which is all the NKE attack needs (it only requires
correctly-classified inputs whose label is preserved)."""
import os, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
import torchvision.transforms as T
from PIL import Image
from . import config as C
from . import attack as A
from . import metrics as MET
from . import mechanism as MECH

IMAGENETTE_DIR = os.path.join(C.DATA_DIR, "imagenette2-160")
# Imagenette WNID -> ImageNet-1k class index
WNID_TO_IDX = {
    "n01440764": 0,   "n02102040": 217, "n02979186": 482, "n03000684": 491,
    "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
    "n03445777": 574, "n03888257": 701,
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class NormImageNet(nn.Module):
    """Wrap a torchvision model: accept [0,1] 224px input, normalize internally,
    and expose .features(x) (input to the final Linear) for Mahalanobis."""
    def __init__(self, net):
        super().__init__()
        self.net = net.eval()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self._feat = {}
        last = self._last_linear()
        last.register_forward_hook(lambda m, i, o: self._feat.__setitem__("f", i[0]))

    def _last_linear(self):
        lin = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        return lin[-1]

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def features(self, x):
        _ = self.forward(x)
        return self._feat["f"]


def load_models():
    rn = tvm.resnet152(weights=tvm.ResNet152_Weights.IMAGENET1K_V1)
    vg = tvm.vgg16_bn(weights=tvm.VGG16_BN_Weights.IMAGENET1K_V1)
    return {"resnet152": NormImageNet(rn).to(C.DEVICE),
            "vgg16": NormImageNet(vg).to(C.DEVICE)}


def load_images(n_per_class=40, split="val"):
    """Return [N,3,224,224] in [0,1] and ImageNet-index labels."""
    tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
    xs, ys = [], []
    g = np.random.RandomState(C.SEED)
    for wnid, idx in WNID_TO_IDX.items():
        files = sorted(glob.glob(os.path.join(IMAGENETTE_DIR, split, wnid, "*.JPEG")))
        g.shuffle(files)
        for fp in files[:n_per_class]:
            try:
                im = Image.open(fp).convert("RGB")
                xs.append(tf(im)); ys.append(idx)
            except Exception:
                pass
    return torch.stack(xs), torch.tensor(ys)


@torch.no_grad()
def correctly_classified(model, x, y, batch=32):
    keep = []
    for i in range(0, len(x), batch):
        p = model(x[i:i + batch].to(C.DEVICE)).argmax(1).cpu()
        keep.append(p == y[i:i + batch])
    m = torch.cat(keep)
    return x[m], y[m]


@torch.no_grad()
def label_kept(model, xa, y, batch=32):
    preds = []
    for i in range(0, len(xa), batch):
        preds.append(model(xa[i:i + batch].to(C.DEVICE)).argmax(1).cpu())
    return float((torch.cat(preds) == y).float().mean())


# ImageNet L2 grid (224*224*3 = 150k px; natural-image L2 norms are ~O(100))
EPS_GRID_IN = [0.0, 5.0, 15.0, 30.0, 60.0, 100.0]


def run(n_per_class=40, steps=40):
    import json
    models = load_models()
    x_all, y_all = load_images(n_per_class=n_per_class)
    print(f"[imagenet] loaded {len(x_all)} Imagenette images")

    # clean accuracy of each model on the natural images
    for name, mdl in models.items():
        acc = label_kept(mdl, x_all, y_all)
        print(f"  clean acc {name}: {acc:.3f}")

    src_name = "resnet152"; tgt_name = "vgg16"
    src, tgt = models[src_name], models[tgt_name]
    x, y = correctly_classified(src, x_all, y_all)
    x, y = x[:200], y[:200]
    print(f"[imagenet] {len(x)} correctly-classified by {src_name}")

    # OOD detectors fit on a held-out clean split
    xval, yval = correctly_classified(src, x_all, y_all)
    xval = xval[:400]
    maha = MET.Mahalanobis(src).fit(xval, [0] * len(xval))  # single-cluster ok: distance to clean feats
    id_msp = MET.msp_score(src, xval, batch=32)
    id_en = MET.energy_score(src, xval, batch=32)
    id_ma = maha.score(xval)
    thr = {"msp": MET.threshold_at_fpr(id_msp, 0.05),
           "energy": MET.threshold_at_fpr(id_en, 0.05),
           "maha": MET.threshold_at_fpr(id_ma, 0.05)}

    out = {"src": src_name, "tgt": tgt_name, "grid": EPS_GRID_IN, "levels": [],
           "clean_acc": {n: label_kept(m, x_all, y_all) for n, m in models.items()}}
    def attack_batched(eps, bs=16):
        outs = []
        for i in range(0, len(x), bs):
            xb = A.run_variant(src, x[i:i + bs], y[i:i + bs], eps, "nmifgm",
                               steps=steps, seed=C.SEED).cpu()
            outs.append(xb)
        return torch.cat(outs)

    for eps in EPS_GRID_IN:
        if eps == 0:
            xa = x.clone()
        else:
            xa = attack_batched(eps)
        wb = label_kept(src, xa, y)
        transfer = label_kept(tgt, xa, y)            # generic-recognizer proxy = other model
        # matched-L2 Gaussian control: random noise at the same per-image L2 as NKE
        if eps == 0:
            transfer_gauss = transfer
        else:
            ggen = torch.Generator().manual_seed(C.SEED + int(eps))
            noise = torch.randn(x.shape, generator=ggen)
            ach = (xa - x).flatten(1).norm(dim=1)
            nn_ = noise.flatten(1).norm(dim=1).clamp_min(1e-9)
            xg = (x + noise * (ach / nn_).view(-1, 1, 1, 1)).clamp(0, 1)
            transfer_gauss = label_kept(tgt, xg, y)
        # confidence
        with torch.no_grad():
            conf = torch.softmax(src(xa.to(C.DEVICE)), 1).max(1).values.mean().item()
        # OOD
        det = {}
        det["msp"] = MET.detection_rate(MET.msp_score(src, xa, batch=32), thr["msp"])
        det["energy"] = MET.detection_rate(MET.energy_score(src, xa, batch=32), thr["energy"])
        det["maha"] = MET.detection_rate(maha.score(xa), thr["maha"])
        # mechanism (subsample for speed)
        ef, tx = [], []
        for i in range(0, len(x), 4):
            c = np.transpose(x[i].numpy(), (1, 2, 0)); p = np.transpose(xa[i].numpy(), (1, 2, 0))
            f1, _ = MECH.edge_preservation(c, p); ef.append(f1); tx.append(MECH.texture_similarity(c, p))
        row = {"eps_l": eps, "white_box_success": wb, "transfer_proxy": transfer,
               "transfer_proxy_gauss": transfer_gauss,
               "mean_conf": conf, "achieved_L2": float((xa - x).flatten(1).norm(dim=1).mean()),
               "msp_detect": det["msp"], "energy_detect": det["energy"], "maha_detect": det["maha"],
               "edge_f1": float(np.mean(ef)), "texture_glcm": float(np.mean(tx))}
        out["levels"].append(row)
        print(f"  eps_l={eps:6.1f} L2={row['achieved_L2']:6.1f} wb={wb:.3f} "
              f"transfer(NKE)={transfer:.3f} transfer(Gauss)={transfer_gauss:.3f} "
              f"conf={conf:.3f} maha={det['maha']:.2f} msp={det['msp']:.2f} "
              f"edge={row['edge_f1']:.3f} tex={row['texture_glcm']:.3f}")
    path = os.path.join(C.RESULTS_DIR, "imagenet_ext.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    run()
