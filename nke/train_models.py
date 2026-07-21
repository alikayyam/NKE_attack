"""Train the clean classifiers (2 archs per dataset) and, optionally, an
adversarially-trained (Madry PGD) CIFAR ResNet for the defense track."""
import os, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as C
from . import data as D
from . import models as M
from .attack import classical_pgd

torch.manual_seed(C.SEED)


def evaluate(model, loader):
    model.eval(); correct = tot = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(C.DEVICE); y = y.to(C.DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); tot += y.numel()
    return correct / tot


def train_one(dataset, arch, epochs, lr, adv=False, eps=8/255):
    tag = "advtrain" if adv else "clean"
    print(f"[train] {dataset}/{arch} tag={tag} epochs={epochs}")
    model = M.build(dataset, arch)
    aug = dataset == "cifar10"
    tr = D.get_loader(dataset, train=True, batch_size=128, augment=aug, num_workers=4)
    te = D.get_loader(dataset, train=False, batch_size=256, num_workers=4)
    warmup = 0
    label_smooth = 0.0
    if adv:
        # adversarial training is unstable under high-LR SGD; use Adam.
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    elif arch == "vit":
        # ViTs need AdamW + weight decay + LR warmup + label smoothing to train
        # from scratch on small data; SGD@0.1 diverges.
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
        warmup = 5
        label_smooth = 0.1
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warmup))
    else:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    base_lr = lr
    for ep in range(epochs):
        model.train(); t0 = time.time()
        if warmup and ep < warmup:  # linear LR warmup
            for g in opt.param_groups:
                g["lr"] = base_lr * (ep + 1) / warmup
        for x, y in tr:
            x = x.to(C.DEVICE); y = y.to(C.DEVICE)
            if adv:
                x = classical_pgd(model, x, y, eps=eps, steps=7)
                model.train()
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y, label_smoothing=label_smooth)
            loss.backward(); opt.step()
        if not (warmup and ep < warmup):
            sched.step()
        if ep == epochs - 1 or ep % 5 == 0:
            print(f"  ep{ep} acc={evaluate(model, te):.4f} ({time.time()-t0:.1f}s)")
    acc = evaluate(model, te)
    path = os.path.join(C.CKPT_DIR, f"{dataset}_{arch}_{tag}.pt")
    torch.save(model.state_dict(), path)
    print(f"[saved] {path} test_acc={acc:.4f}")
    return acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="e.g. cifar10:resnet18")
    ap.add_argument("--advtrain", action="store_true")
    args = ap.parse_args()

    jobs = [
        ("mnist", "lenet", 5, 0.05),
        ("mnist", "mlp", 8, 0.05),
        ("cifar10", "resnet18", 30, 0.1),
        ("cifar10", "vgg11", 30, 0.1),
        ("cifar10", "vit", 70, 5e-4),
    ]
    results = {}
    for ds, arch, ep, lr in jobs:
        if args.only and f"{ds}:{arch}" != args.only:
            continue
        results[f"{ds}_{arch}"] = train_one(ds, arch, ep, lr)
    if args.advtrain:
        results["cifar10_resnet18_adv"] = train_one("cifar10", "resnet18", 30, 0.1, adv=True)
    print("SUMMARY", results)
