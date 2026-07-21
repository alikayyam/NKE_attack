"""CLIP zero-shot recognizability proxy. CLIP is a stronger, broadly-trained
'generic recognizer' that is NOT the attack's transfer target, so it is a better
stand-in for human recognizability than a second task-specific CNN -- especially
at ImageNet scale, where NKE examples are highly model-specific and a same-task
CNN proxy collapses to the transfer rate. If CLIP still recognizes NKE images
where a second CNN does not, the 'not signal loss' effect survives on firmer
ground; if CLIP also collapses (to the level of a Gaussian control), it does not."""
import os
import numpy as np
import torch
import torch.nn.functional as F
import open_clip
from . import config as C

_CLIP = {}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PROMPTS = {
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog",
                "horse", "ship", "truck"],
    "imagenet": ["tench", "English springer dog", "cassette player", "chain saw",
                 "church", "French horn", "garbage truck", "gas pump", "golf ball",
                 "parachute"],
}
# imagenet label list index -> our prompt index
IMAGENET_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]


def _load_clip():
    if "m" not in _CLIP:
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
        _CLIP["m"] = model.to(C.DEVICE).eval()
        _CLIP["tok"] = open_clip.get_tokenizer("ViT-B-32")
    return _CLIP["m"], _CLIP["tok"]


@torch.no_grad()
def _text_feats(classes):
    model, tok = _load_clip()
    txt = tok([f"a photo of a {c}" for c in classes]).to(C.DEVICE)
    f = model.encode_text(txt); f /= f.norm(dim=-1, keepdim=True)
    return f


@torch.no_grad()
def clip_zeroshot_acc(images01, labels_promptidx, classes, batch=64):
    """images01: [N,C,H,W] in [0,1]; labels_promptidx: index into `classes`."""
    model, _ = _load_clip()
    tf = _text_feats(classes)
    mean = torch.tensor(CLIP_MEAN, device=C.DEVICE).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=C.DEVICE).view(1, 3, 1, 1)
    correct = 0
    for i in range(0, len(images01), batch):
        x = images01[i:i + batch].to(C.DEVICE)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        f = model.encode_image(x); f /= f.norm(dim=-1, keepdim=True)
        pred = (f @ tf.T).argmax(1).cpu()
        correct += (pred == labels_promptidx[i:i + batch]).sum().item()
    return correct / len(images01)


def run_cifar():
    ss = torch.load(os.path.join(C.RESULTS_DIR, "stimset_cifar10_resnet18_nmifgm_n200.pt"))
    y = ss["y"]  # already 0..9 matching PROMPTS order
    classes = PROMPTS["cifar10"]
    out = {"dataset": "cifar10", "levels": []}
    for eps in ss["grid"]:
        acc_nke = clip_zeroshot_acc(ss["levels"][eps]["nke"], y, classes)
        acc_g = clip_zeroshot_acc(ss["levels"][eps]["gauss"], y, classes)
        out["levels"].append({"eps_l": eps, "clip_nke": acc_nke, "clip_gauss": acc_g,
                              "model_acc": ss["levels"][eps]["model_acc"]})
        print(f"  cifar eps_l={eps:5.1f} model=1.00 clip(nke)={acc_nke:.3f} clip(gauss)={acc_g:.3f}")
    import json
    json.dump(out, open(os.path.join(C.RESULTS_DIR, "clip_cifar10.json"), "w"), indent=2)
    return out


def run_imagenet():
    from . import imagenet_ext as IE
    from . import attack as A
    models = IE.load_models(); src = models["resnet152"]
    x_all, y_all = IE.load_images(n_per_class=40)
    x, y = IE.correctly_classified(src, x_all, y_all); x, y = x[:200], y[:200]
    classes = PROMPTS["imagenet"]
    remap = {idx: i for i, idx in enumerate(IMAGENET_IDX)}
    y_prompt = torch.tensor([remap[int(v)] for v in y])
    out = {"dataset": "imagenet", "levels": []}
    for eps in IE.EPS_GRID_IN:
        if eps == 0:
            xa = x.clone(); xg = x.clone()
        else:
            xa = torch.cat([A.run_variant(src, x[i:i+16], y[i:i+16], eps, "nmifgm", steps=40, seed=C.SEED).cpu()
                            for i in range(0, len(x), 16)])
            gg = torch.Generator().manual_seed(C.SEED + int(eps))
            noise = torch.randn(x.shape, generator=gg)
            ach = (xa - x).flatten(1).norm(dim=1); nn_ = noise.flatten(1).norm(dim=1).clamp_min(1e-9)
            xg = (x + noise * (ach / nn_).view(-1, 1, 1, 1)).clamp(0, 1)
        acc_nke = clip_zeroshot_acc(xa, y_prompt, classes)
        acc_g = clip_zeroshot_acc(xg, y_prompt, classes)
        out["levels"].append({"eps_l": eps, "clip_nke": acc_nke, "clip_gauss": acc_g})
        print(f"  imagenet eps_l={eps:6.1f} clip(nke)={acc_nke:.3f} clip(gauss)={acc_g:.3f}")
    import json
    json.dump(out, open(os.path.join(C.RESULTS_DIR, "clip_imagenet.json"), "w"), indent=2)
    return out


if __name__ == "__main__":
    print("=== CIFAR-10 ==="); run_cifar()
    print("=== ImageNet ==="); run_imagenet()
