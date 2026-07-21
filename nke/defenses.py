"""Defense wrappers evaluated in Sec 3.5. Each wraps a loaded NormalizedModel
(which takes [0,1] input) and is itself callable as model(x)->logits, so the
same nke_attack runs white-box against it. Non-differentiable transforms use a
straight-through (BPDA) backward."""
import io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from . import config as C


class _JPEGFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        arr = x.detach().cpu().clamp(0, 1).numpy()
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            im = arr[i]
            chw = im.shape[0]
            a = (np.transpose(im, (1, 2, 0)) * 255).astype(np.uint8) if chw == 3 else (im[0] * 255).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(a).save(buf, format="JPEG", quality=int(quality))
            dec = np.asarray(Image.open(buf)).astype(np.float32) / 255.0
            out[i] = np.transpose(dec, (2, 0, 1)) if chw == 3 else dec[None]
        return torch.from_numpy(out).to(x.device)

    @staticmethod
    def backward(ctx, g):
        return g, None  # straight-through


class JPEGDefense(nn.Module):
    def __init__(self, model, quality=40):
        super().__init__(); self.model = model; self.q = quality

    def forward(self, x):
        return self.model(_JPEGFn.apply(x, self.q))


def _gauss_kernel(sigma, chans, device):
    r = max(1, int(3 * sigma)); xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k1 = torch.exp(-(xs ** 2) / (2 * sigma ** 2)); k1 /= k1.sum()
    k2 = torch.outer(k1, k1)
    return k2.view(1, 1, *k2.shape).repeat(chans, 1, 1, 1).to(device), r


class BlurDefense(nn.Module):
    def __init__(self, model, sigma=1.0, chans=3):
        super().__init__(); self.model = model
        self.k, self.r = _gauss_kernel(sigma, chans, C.DEVICE); self.chans = chans

    def forward(self, x):
        xb = F.conv2d(F.pad(x, [self.r] * 4, mode="reflect"), self.k, groups=self.chans)
        return self.model(xb)


class RandomResizePadDefense(nn.Module):
    """Guo et al. random resize + random pad back to original size (differentiable)."""
    def __init__(self, model, size, max_scale=1.18):
        super().__init__(); self.model = model; self.size = size; self.max_scale = max_scale

    def forward(self, x):
        H = self.size
        rs = int(H * (1 + torch.rand(1).item() * (self.max_scale - 1)))
        xr = F.interpolate(x, size=(rs, rs), mode="bilinear", align_corners=False)
        pad = H * 2 - rs if rs < H * 2 else 0
        total = max(0, (H - rs))
        if rs <= H:
            l = int(torch.randint(0, (H - rs) + 1, (1,)).item()); r = H - rs - l
            t = int(torch.randint(0, (H - rs) + 1, (1,)).item()); b = H - rs - t
            xr = F.pad(xr, [l, r, t, b])
        else:
            xr = F.interpolate(xr, size=(H, H), mode="bilinear", align_corners=False)
        return self.model(xr)


class SmoothingDefense(nn.Module):
    """Randomized smoothing (Cohen et al.): average logits over k Gaussian draws.
    Differentiable (EOT), so the attacker gets a smoothed-surrogate gradient."""
    def __init__(self, model, sigma=0.25, k=8):
        super().__init__(); self.model = model; self.sigma = sigma; self.k = k

    def forward(self, x):
        acc = 0
        for _ in range(self.k):
            acc = acc + self.model((x + torch.randn_like(x) * self.sigma).clamp(0, 1))
        return acc / self.k


def build_defense(name, model, dataset):
    chans = C.IN_CHANS[dataset]; size = 28 if dataset == "mnist" else 32
    if name == "jpeg":
        return JPEGDefense(model, quality=40 if dataset == "cifar10" else 50)
    if name == "blur":
        return BlurDefense(model, sigma=1.0, chans=chans)
    if name == "resize_pad":
        return RandomResizePadDefense(model, size=size)
    if name == "smoothing":
        return SmoothingDefense(model, sigma=0.25, k=8)
    raise ValueError(name)
