"""Adaptive attack vs. the Mahalanobis detector (the one detector that catches
NKE). The adaptive attacker minimises  CE(f(x'),y) + lambda * Maha_feat(x')
--- keeping the label AND pulling the penultimate feature back toward the clean
class manifold to evade Mahalanobis --- while still projecting the perturbation
to L2 distance >= eps_l. If detection can be driven down while success stays high
and the perturbation stays large, even feature-distance OOD detection fails
against an adaptive adversary; if not, that detection is robust."""
import json, os
import numpy as np
import torch
import torch.nn.functional as F
from . import config as C
from . import data as D
from . import models as M
from . import metrics as MET
from .attack import _pnorm, _project_lower_bound, _clip01


class TorchMaha:
    """Differentiable tied-covariance Mahalanobis over a chosen feature map
    (default: the model's penultimate features)."""
    def __init__(self, model, feat_fn=None):
        self.model = model
        self.feat_fn = feat_fn or (lambda x: model.features(x))

    @torch.no_grad()
    def fit(self, x, y, batch=256):
        feats = []
        for i in range(0, len(x), batch):
            feats.append(self.feat_fn(x[i:i + batch].to(C.DEVICE)))
        Fm = torch.cat(feats); y = y.to(C.DEVICE)
        self.classes = torch.unique(y)
        d = Fm.shape[1]; cov = torch.zeros(d, d, device=C.DEVICE); self.means = {}
        for c in self.classes:
            fc = Fm[y == c]; mu = fc.mean(0); self.means[int(c)] = mu
            cov += (fc - mu).T @ (fc - mu)
        cov /= len(Fm)
        self.prec = torch.linalg.pinv(cov + 1e-6 * torch.eye(d, device=C.DEVICE))
        self.mu = torch.stack([self.means[int(c)] for c in self.classes])  # [K,d]
        return self

    def dist(self, feats):
        """min-over-class squared Mahalanobis distance, differentiable. [N]"""
        diff = feats.unsqueeze(1) - self.mu.unsqueeze(0)      # [N,K,d]
        m = torch.einsum("nkd,de,nke->nk", diff, self.prec, diff)
        return m.min(1).values


class TorchMahaUntied:
    """Differentiable per-class-covariance Mahalanobis over a chosen feature map
    (matches the deployed untied `_MahaNP(tied=False)` detector below)."""
    def __init__(self, model, feat_fn=None):
        self.model = model
        self.feat_fn = feat_fn or (lambda x: model.features(x))

    @torch.no_grad()
    def fit(self, x, y, batch=256):
        feats = []
        for i in range(0, len(x), batch):
            feats.append(self.feat_fn(x[i:i + batch].to(C.DEVICE)))
        Fm = torch.cat(feats); y = y.to(C.DEVICE)
        self.classes = torch.unique(y); d = Fm.shape[1]
        means, precs = [], []
        for c in self.classes:
            fc = Fm[y == c]; mu = fc.mean(0); means.append(mu)
            cov = (fc - mu).T @ (fc - mu) / max(1, len(fc) - 1)
            precs.append(torch.linalg.pinv(cov + 1e-6 * torch.eye(d, device=C.DEVICE)))
        self.mu = torch.stack(means)          # [K,d]
        self.prec = torch.stack(precs)        # [K,d,d]
        return self

    def dist(self, feats):
        diff = feats.unsqueeze(1) - self.mu.unsqueeze(0)      # [N,K,d]
        m = torch.einsum("nkd,kde,nke->nk", diff, self.prec, diff)
        return m.min(1).values


def adaptive_attack(model, tmaha, x, y, eps_l, lam, steps=60, alpha=None, seed=C.SEED):
    model.eval()
    x = x.to(C.DEVICE); y = y.to(C.DEVICE)
    if alpha is None:
        alpha = 2.5 * eps_l / steps
    shape = [-1] + [1] * (x.dim() - 1)
    g = torch.Generator(device=C.DEVICE).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=C.DEVICE)
    x_adv = _clip01(x + noise * (eps_l / _pnorm(noise, 2).clamp_min(1e-9)).view(*shape)).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        feats = model.features(x_adv)
        ce = F.cross_entropy(logits, y)
        maha = tmaha.dist(feats).mean()
        loss = ce + lam * maha                     # minimise both
        grad = torch.autograd.grad(loss, x_adv)[0]
        gn = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(*shape)
        x_adv = (x_adv - alpha * grad / gn).detach()
        x_adv = _project_lower_bound(x_adv, x, eps_l, 2)
        x_adv = _clip01(x_adv)
    return x_adv.detach()


def run(dataset, arch, n=200, tag="clean"):
    model = M.load(dataset, arch, tag)
    x, y = D.sample_correct(model, dataset, n)
    from .attackset import EPS_GRID
    eps_l = EPS_GRID[dataset][-2]        # a large level where standard NKE is 100% detected

    # Split a fit pool into two DISJOINT halves so the held-out detector is fit on
    # data the attacker never sees:
    #   half A -> the attacker's (differentiable) detector + a white-box deployed detector
    #   half B -> a held-out deployed detector, statistically independent of the attack
    xpool, ypool = D.sample_correct(model, dataset, 4000, seed=C.SEED + 7)
    half = len(xpool) // 2
    xa_fit, ya_fit = xpool[:half], ypool[:half]        # attacker sees this
    xb_fit, yb_fit = xpool[half:], ypool[half:]        # held out from attacker

    tmaha = TorchMaha(model).fit(xa_fit, ya_fit)       # what the adaptive attacker optimises against

    # 5%-FPR thresholds are calibrated on a DISJOINT clean split (not the fit set):
    # a fitted-covariance detector overfits, so an in-sample threshold understates
    # the true FPR.
    xcal, _ = D.sample_correct(model, dataset, 2000, seed=C.SEED + 123)
    # (1) white-box deployed detector: exactly the one the attacker differentiated through
    maha_wb = MET.Mahalanobis(model).fit(xa_fit, ya_fit.numpy())
    thr_wb = MET.threshold_at_fpr(maha_wb.score(xcal), 0.05)
    # (2) held-out deployed detector: same estimator, disjoint fit data
    maha_ho = MET.Mahalanobis(model).fit(xb_fit, yb_fit.numpy())
    thr_ho = MET.threshold_at_fpr(maha_ho.score(xcal), 0.05)
    thr_msp = MET.threshold_at_fpr(MET.msp_score(model, xcal), 0.05)

    @torch.no_grad()
    def kept(xa):
        return float((model(xa.to(C.DEVICE)).argmax(1).cpu() == y).float().mean())

    out = {"dataset": dataset, "arch": arch, "eps_l": eps_l, "lambdas": []}
    # refined grid: resolve the success cliff between 0.01 (fully evasive) and 0.1 (collapsed)
    for lam in [0.0, 0.001, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 1.0]:
        xa = adaptive_attack(model, tmaha, x, y, eps_l, lam).cpu()
        row = {"lambda": lam, "success": kept(xa),
               "achieved_L2": float((xa - x).flatten(1).norm(dim=1).mean()),
               "maha_detect": MET.detection_rate(maha_wb.score(xa), thr_wb),
               "maha_detect_heldout": MET.detection_rate(maha_ho.score(xa), thr_ho),
               "msp_detect": MET.detection_rate(MET.msp_score(model, xa), thr_msp)}
        out["lambdas"].append(row)
        print(f"  lambda={lam:6.3f} success={row['success']:.3f} L2={row['achieved_L2']:5.2f} "
              f"maha_wb={row['maha_detect']:.3f} maha_heldout={row['maha_detect_heldout']:.3f} "
              f"msp={row['msp_detect']:.3f}")
    out["tag"] = tag
    suffix = "" if tag == "clean" else f"_{tag}"
    path = os.path.join(C.RESULTS_DIR, f"adaptive_{dataset}_{arch}{suffix}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    return out


# ======================= strong / different detectors =======================
# Does evasion of the single tied-covariance Mahalanobis detector (penultimate
# layer) that the attacker targets also fool detectors it did NOT target ---
# per-class (untied) covariance, Mahalanobis at a different layer, and a
# non-parametric feature-space kNN detector (Sun et al. 2022)? And can an
# attacker that penalises a multi-layer Mahalanobis *ensemble* evade all of them
# at once? All detectors are fit on the same reference features and thresholded
# at 5% FPR, exactly as in the OOD track.

def _layer3_feat_fn(model, arch):
    """A genuinely different feature layer for a 'different-layer' detector.
    Implemented for resnet18 (post-layer3 global-avg-pool, 256-d)."""
    if arch != "resnet18":
        return None
    net = model.net

    def f(x):
        o = torch.nn.functional.relu(net.bn1(net.conv1(model.norm(x))))
        o = net.layer1(o); o = net.layer2(o); o = net.layer3(o)
        o = torch.nn.functional.adaptive_avg_pool2d(o, 1)
        return torch.flatten(o, 1)
    return f


@torch.no_grad()
def _extract(feat_fn, x, batch=256):
    out = []
    for i in range(0, len(x), batch):
        out.append(feat_fn(x[i:i + batch].to(C.DEVICE)).cpu())
    return torch.cat(out).numpy()


class _MahaNP:
    """Numpy Mahalanobis on precomputed features; tied or per-class covariance."""
    def __init__(self, tied=True):
        self.tied = tied

    def fit(self, Fm, y):
        y = np.asarray(y); self.classes = np.unique(y); self.means = {}; self.prec = {}
        d = Fm.shape[1]
        if self.tied:
            cov = np.zeros((d, d))
            for c in self.classes:
                fc = Fm[y == c]; mu = fc.mean(0); self.means[c] = mu
                cov += (fc - mu).T @ (fc - mu)
            cov /= len(Fm); P = np.linalg.pinv(cov + 1e-6 * np.eye(d))
            self.prec = {c: P for c in self.classes}
        else:
            for c in self.classes:
                fc = Fm[y == c]; mu = fc.mean(0); self.means[c] = mu
                cov = (fc - mu).T @ (fc - mu) / max(1, len(fc) - 1)
                self.prec[c] = np.linalg.pinv(cov + 1e-6 * np.eye(d))
        return self

    def score(self, Fm):
        ds = [np.einsum("ni,ij,nj->n", Fm - self.means[c], self.prec[c], Fm - self.means[c])
              for c in self.classes]
        return np.min(np.stack(ds, 1), 1)


class _KNN:
    """Non-parametric feature-space kNN OOD score (Sun et al. 2022): negative
    cosine similarity to the k-th nearest training feature."""
    def __init__(self, k=50):
        self.k = k

    def fit(self, Fm, y=None):
        self.bank = Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-12)
        return self

    def score(self, Fm):
        q = Fm / (np.linalg.norm(Fm, axis=1, keepdims=True) + 1e-12)
        sims = q @ self.bank.T
        kth = np.partition(sims, -self.k, axis=1)[:, -self.k]
        return -kth


def _adaptive_attack_pen(model, penalty_fn, x, y, eps_l, lam, steps=60, alpha=None, seed=C.SEED):
    """Same as adaptive_attack but the feature penalty is an arbitrary callable
    penalty_fn(x_adv)->scalar (used for the multi-layer ensemble attacker)."""
    model.eval()
    x = x.to(C.DEVICE); y = y.to(C.DEVICE)
    if alpha is None:
        alpha = 2.5 * eps_l / steps
    shape = [-1] + [1] * (x.dim() - 1)
    g = torch.Generator(device=C.DEVICE).manual_seed(seed)
    noise = torch.randn(x.shape, generator=g, device=C.DEVICE)
    x_adv = _clip01(x + noise * (eps_l / _pnorm(noise, 2).clamp_min(1e-9)).view(*shape)).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y) + lam * penalty_fn(x_adv)
        grad = torch.autograd.grad(loss, x_adv)[0]
        gn = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(*shape)
        x_adv = (x_adv - alpha * grad / gn).detach()
        x_adv = _project_lower_bound(x_adv, x, eps_l, 2)
        x_adv = _clip01(x_adv)
    return x_adv.detach()


def run_strong_detectors(dataset, arch, n=200):
    model = M.load(dataset, arch, "clean")
    x, y = D.sample_correct(model, dataset, n)
    from .attackset import EPS_GRID
    eps_l = EPS_GRID[dataset][-2]
    xref, yref = D.sample_correct(model, dataset, 2000, seed=C.SEED + 7)

    feat_penult = lambda z: model.features(z)
    feat_l3 = _layer3_feat_fn(model, arch)

    # deployed detector battery (fit on reference features, threshold at 5% FPR)
    dets = {}
    Fp = _extract(feat_penult, xref)
    dets["tied_penult"]   = (_MahaNP(True).fit(Fp, yref.numpy()), feat_penult)
    dets["untied_penult"] = (_MahaNP(False).fit(Fp, yref.numpy()), feat_penult)
    dets["knn_penult"]    = (_KNN(50).fit(Fp), feat_penult)
    if feat_l3 is not None:
        Fl3 = _extract(feat_l3, xref)
        dets["tied_layer3"] = (_MahaNP(True).fit(Fl3, yref.numpy()), feat_l3)
    # Calibrate each 5%-FPR threshold on a DISJOINT clean split (not the fit set),
    # then verify the true FPR on a THIRD disjoint clean split. A fitted-covariance
    # detector overfits, so an in-sample threshold badly understates its real FPR
    # (e.g. untied per-class covariance: ~77% real FPR at an in-sample "5%").
    xcal, _ = D.sample_correct(model, dataset, 2000, seed=C.SEED + 123)
    xtest, _ = D.sample_correct(model, dataset, 1000, seed=C.SEED + 321)
    thr = {name: MET.threshold_at_fpr(det.score(_extract(ff, xcal)), 0.05)
           for name, (det, ff) in dets.items()}
    heldout_fpr = {name: MET.detection_rate(det.score(_extract(ff, xtest)), thr[name])
                   for name, (det, ff) in dets.items()}
    print("held-out clean FPR (calibrated on disjoint split):",
          {k: round(v, 3) for k, v in heldout_fpr.items()})

    # differentiable penalties the attacker can target
    tm_penult = TorchMaha(model, feat_penult).fit(xref, yref)
    tm_untied = TorchMahaUntied(model, feat_penult).fit(xref, yref)
    pen_penult = lambda z: tm_penult.dist(feat_penult(z)).mean()
    pen_untied = lambda z: tm_untied.dist(feat_penult(z)).mean()   # target untied alone
    if feat_l3 is not None:
        tm_l3 = TorchMaha(model, feat_l3).fit(xref, yref)
        pen_ensemble = lambda z: tm_penult.dist(feat_penult(z)).mean() + tm_l3.dist(feat_l3(z)).mean()
        # kitchen-sink: penalise every parametric detector in the battery at once
        pen_kitchen = lambda z: (tm_penult.dist(feat_penult(z)).mean()
                                 + tm_untied.dist(feat_penult(z)).mean()
                                 + tm_l3.dist(feat_l3(z)).mean())
    else:
        pen_ensemble = None
        pen_kitchen = lambda z: (tm_penult.dist(feat_penult(z)).mean()
                                 + tm_untied.dist(feat_penult(z)).mean())

    @torch.no_grad()
    def kept(xa):
        return float((model(xa.to(C.DEVICE)).argmax(1).cpu() == y).float().mean())

    def measure(xa):
        xa = xa.cpu()
        row = {"success": kept(xa)}
        for name, (det, ff) in dets.items():
            row[name] = MET.detection_rate(det.score(_extract(ff, xa)), thr[name])
        return row

    out = {"dataset": dataset, "arch": arch, "eps_l": eps_l,
           "detectors": list(dets.keys()), "k_knn": 50,
           "heldout_clean_fpr": heldout_fpr,
           "single_target": [], "ensemble_target": [], "kitchen_sink": [],
           "untied_target": []}

    def sweep(name, pen, lams):
        print(f"[{name}]:")
        for lam in lams:
            r = measure(_adaptive_attack_pen(model, pen, x, y, eps_l, lam))
            r["lambda"] = lam; out[name].append(r)
            print(f"  lam={lam:5.3f} succ={r['success']:.3f} " +
                  " ".join(f"{k}={r[k]:.2f}" for k in dets))

    sweep("single_target", pen_penult, [0.0, 0.01, 0.02])       # penalise tied_penult only
    if pen_ensemble is not None:
        sweep("ensemble_target", pen_ensemble, [0.0, 0.01, 0.02])  # + tied_layer3
    # target the untied (per-class covariance) detector ALONE, to check whether the
    # summed kitchen-sink objective merely under-weighted it
    sweep("untied_target", pen_untied, [0.0, 0.001, 0.01, 0.02, 0.05, 0.1])
    # kitchen-sink: penalise tied_penult + untied_penult (+ tied_layer3) jointly.
    # broader lambda grid since three metrics at different scales must all be satisfied
    sweep("kitchen_sink", pen_kitchen, [0.0, 0.01, 0.02, 0.05, 0.1, 0.2])

    path = os.path.join(C.RESULTS_DIR, f"adaptive_detectors_{dataset}_{arch}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"[saved] {path}")
    return out


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("all", "main"):
        run("cifar10", "resnet18")
    if mode in ("all", "advtrain"):
        print("\n===== adversarially-trained ResNet18 =====")
        run("cifar10", "resnet18", tag="advtrain")
    if mode in ("all", "detectors"):
        print("\n===== strong / different detectors =====")
        run_strong_detectors("cifar10", "resnet18")
