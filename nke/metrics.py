"""Shared metrics: calibration (ECE), OOD scores (MSP/energy/Mahalanobis),
and detection-rate helpers."""
import numpy as np
import torch
import torch.nn.functional as F
from . import config as C


# ---------------- calibration ----------------
def expected_calibration_error(confs, correct, n_bins=15):
    confs = np.asarray(confs); correct = np.asarray(correct, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0; N = len(confs)
    reliab = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        if m.sum() == 0:
            reliab.append((0.5 * (lo + hi), np.nan, 0)); continue
        acc = correct[m].mean(); conf = confs[m].mean(); w = m.sum() / N
        ece += w * abs(acc - conf)
        reliab.append((conf, acc, int(m.sum())))
    return float(ece), reliab


# ---------------- OOD scores (higher = more OOD) ----------------
@torch.no_grad()
def msp_score(model, x, batch=256):
    """Negative max-softmax-probability (Hendrycks & Gimpel 2017)."""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(C.DEVICE)
        p = F.softmax(model(xb), 1).max(1).values
        out.append((-p).cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def energy_score(model, x, T=1.0, batch=256):
    """Negative free energy = -T*logsumexp(logits/T) (Liu et al. 2020)."""
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(C.DEVICE)
        e = -T * torch.logsumexp(model(xb) / T, 1)
        out.append(e.cpu())
    return torch.cat(out).numpy()


class Mahalanobis:
    """Mahalanobis-distance detector on penultimate features (Lee et al. 2018).
    Score = min over classes of (f-mu_c)^T Sigma^-1 (f-mu_c)."""
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def _feats(self, x, batch=256):
        fs = []
        for i in range(0, len(x), batch):
            fs.append(self.model.features(x[i:i + batch].to(C.DEVICE)).cpu())
        return torch.cat(fs).numpy()

    def fit(self, x_train, y_train):
        F_ = self._feats(x_train); y = np.asarray(y_train)
        self.classes = np.unique(y)
        self.means = {}
        d = F_.shape[1]; cov = np.zeros((d, d))
        for c in self.classes:
            fc = F_[y == c]; mu = fc.mean(0); self.means[c] = mu
            cov += (fc - mu).T @ (fc - mu)
        cov /= len(F_)
        self.prec = np.linalg.pinv(cov + 1e-6 * np.eye(d))
        return self

    def score(self, x):
        F_ = self._feats(x)
        dists = []
        for c in self.classes:
            diff = F_ - self.means[c]
            dists.append(np.einsum("ni,ij,nj->n", diff, self.prec, diff))
        return np.min(np.stack(dists, 1), 1)


# ---------------- detection rate at fixed FPR ----------------
def threshold_at_fpr(id_scores, fpr=0.05):
    """Threshold s.t. `fpr` fraction of in-distribution scores exceed it."""
    return float(np.quantile(id_scores, 1 - fpr))


def detection_rate(ood_scores, thr):
    return float((np.asarray(ood_scores) > thr).mean())


def auroc(id_scores, ood_scores):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))
