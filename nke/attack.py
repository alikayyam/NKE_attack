"""NKE-style attacks: large, label-preserving perturbations.

Realizes Eq.1 / Nie Eq.7:   min_x' J(x', y_true)   s.t.  D(x, x') >= eps_l

We minimize the true-label loss (keep the model correct) while a lower-bound
projection pushes/holds the perturbation at distance >= eps_l from the clean
image. Variants:
  I-FGSM  (Borji 2022)        : sign gradient descent, no momentum, L_inf/L2 proj
  NI-FGM  (Nie 2025)          : L2-normalized gradient descent
  NMI-FGSM(Nie 2025)          : momentum + sign
  NMI-FGM (Nie 2025)          : momentum + L2-normalized gradient

Also provides classical I-FGSM (small-eps, loss-ascent) for the calibration/OOD
comparison track.
"""
import torch
import torch.nn.functional as F
from . import config as C


def _pnorm(delta, p):
    flat = delta.flatten(1)
    if p == 2:
        return flat.norm(dim=1, p=2)
    elif p == float("inf"):
        return flat.abs().amax(dim=1)
    raise ValueError(p)


def _project_lower_bound(x_adv, x, eps_l, p, sphere=False):
    """Ensure D(x, x_adv) >= eps_l. If inside the ball, push the perturbation
    OUT onto the sphere of radius eps_l (exterior projection). If sphere=True,
    project onto the exact sphere D == eps_l (both bounds) -- used to compare
    attack variants at *equal* perturbation magnitude."""
    delta = x_adv - x
    d = _pnorm(delta, p)
    mask = (d != eps_l) if sphere else (d < eps_l)
    if mask.any():
        scale = torch.ones_like(d)
        safe = d.clamp_min(1e-12)
        scale[mask] = (eps_l / safe)[mask]
        shape = [-1] + [1] * (x.dim() - 1)
        x_adv = x + delta * scale.view(*shape)
    return x_adv


@torch.no_grad()
def _clip01(x):
    return x.clamp(0.0, 1.0)


def nke_attack(model, x, y, eps_l, steps=50, alpha=None, p=2,
               momentum=0.0, sign=True, init="boundary", seed=None, sphere=False):
    """Generate NKE-style adversarial examples at target distance eps_l.

    p            : distance metric (2 or inf)
    momentum     : mu decay (0 => no momentum; NMI variants use >0)
    sign         : True => sign-gradient step (FGSM family); False => L2-normalized (FGM family)
    init         : 'boundary' start x on the eps_l sphere via random direction, then descend loss;
                   'clean' start at x.
    Returns x_adv in [0,1].
    """
    model.eval()
    x = x.to(C.DEVICE); y = y.to(C.DEVICE)
    if alpha is None:
        alpha = (2.5 * eps_l / steps) if not sign else max(eps_l / steps, 1.0 / 255)
    shape = [-1] + [1] * (x.dim() - 1)

    if init == "boundary":
        g = torch.Generator(device=C.DEVICE)
        if seed is not None:
            g.manual_seed(seed)
        noise = torch.randn(x.shape, generator=g, device=C.DEVICE)
        d = _pnorm(noise, p).clamp_min(1e-12)
        x_adv = _clip01(x + noise * (eps_l / d).view(*shape)).detach()
    else:
        x_adv = x.clone().detach()

    grad_mom = torch.zeros_like(x)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = F.cross_entropy(logits, y)          # minimize => keep label correct
        grad = torch.autograd.grad(loss, x_adv)[0]

        if momentum > 0:
            gnorm = grad.flatten(1).abs().sum(1).clamp_min(1e-12).view(*shape)
            grad_mom = momentum * grad_mom + grad / gnorm
            step_dir = grad_mom
        else:
            step_dir = grad

        if sign:
            update = alpha * step_dir.sign()
        else:
            gn = step_dir.flatten(1).norm(dim=1).clamp_min(1e-12).view(*shape)
            update = alpha * step_dir / gn

        x_adv = (x_adv - update).detach()           # descent on loss
        x_adv = _project_lower_bound(x_adv, x, eps_l, p, sphere=sphere)
        x_adv = _clip01(x_adv)
    return x_adv.detach()


VARIANTS = {
    # name        -> kwargs
    "ifgsm":    dict(p=float("inf"), momentum=0.0, sign=True),   # Borji L_inf
    "nifgm":    dict(p=2,            momentum=0.0, sign=False),  # Nie L2
    "nmifgsm":  dict(p=float("inf"), momentum=1.0, sign=True),   # Nie momentum+sign
    "nmifgm":   dict(p=2,            momentum=1.0, sign=False),  # Nie momentum+L2
}


def run_variant(model, x, y, eps_l, variant, steps=50, **kw):
    cfg = dict(VARIANTS[variant]); cfg.update(kw)
    return nke_attack(model, x, y, eps_l, steps=steps, **cfg)


# ---------- classical small-eps adversarial attack (for OOD/calibration comparison) ----------
def classical_pgd(model, x, y, eps=8 / 255, steps=20, alpha=None, p=float("inf")):
    """Standard loss-ASCENT PGD producing a *wrong* prediction, bounded above by eps."""
    model.eval()
    x = x.to(C.DEVICE); y = y.to(C.DEVICE)
    if alpha is None:
        alpha = 2.5 * eps / steps
    shape = [-1] + [1] * (x.dim() - 1)
    x_adv = (x + (torch.rand_like(x) * 2 - 1) * eps).clamp(0, 1).detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        if p == float("inf"):
            x_adv = x_adv + alpha * grad.sign()
            delta = (x_adv - x).clamp(-eps, eps)
        else:
            gn = grad.flatten(1).norm(dim=1).clamp_min(1e-12).view(*shape)
            x_adv = x_adv + alpha * grad / gn
            delta = x_adv - x
            dn = _pnorm(delta, 2)
            factor = (eps / dn.clamp_min(1e-12)).clamp(max=1.0)
            delta = delta * factor.view(*shape)
        x_adv = (x + delta).clamp(0, 1).detach()
    return x_adv.detach()
