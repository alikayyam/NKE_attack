"""Model definitions + a NormalizedModel wrapper so attacks operate in [0,1].

Two architecturally distinct families per dataset (mirroring Nie et al.'s choice
of distinct families for the cross-model transfer matrix):
  MNIST:    LeNet-style CNN  vs  MLP
  CIFAR-10: ResNet-18        vs  VGG-11-style CNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config as C


class NormalizedModel(nn.Module):
    """Wrap a network so it accepts [0,1] input and normalizes internally.
    Exposes .features(x) for the penultimate representation (Mahalanobis)."""
    def __init__(self, net, mean, std):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def norm(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.net(self.norm(x))

    def features(self, x):
        return self.net.features_forward(self.norm(x))


# ---------------- MNIST ----------------
class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def features_forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        return self.fc2(self.features_forward(x))


class MLP(nn.Module):
    def __init__(self, num_classes=10, in_ch=1):
        super().__init__()
        self.in_dim = in_ch * 28 * 28
        self.fc1 = nn.Linear(self.in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def features_forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x):
        return self.fc3(self.features_forward(x))


# ---------------- CIFAR-10 ----------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        self.short = nn.Sequential()
        if stride != 1 or inp != out:
            self.short = nn.Sequential(nn.Conv2d(inp, out, 1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out))

    def forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        return F.relu(o + self.short(x))


class ResNet18(nn.Module):
    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        self.inp = 64
        self.conv1 = nn.Conv2d(in_ch, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 2, 1)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _make(self, out, n, stride):
        strides = [stride] + [1] * (n - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.inp, out, s)); self.inp = out
        return nn.Sequential(*layers)

    def features_forward(self, x):
        o = F.relu(self.bn1(self.conv1(x)))
        o = self.layer1(o); o = self.layer2(o); o = self.layer3(o); o = self.layer4(o)
        o = F.adaptive_avg_pool2d(o, 1)
        return torch.flatten(o, 1)

    def forward(self, x):
        return self.fc(self.features_forward(x))


class VGG11(nn.Module):
    cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

    def __init__(self, num_classes=10, in_ch=3):
        super().__init__()
        layers = []; c = in_ch
        for v in self.cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(c, v, 3, padding=1), nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                c = v
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, num_classes)

    def features_forward(self, x):
        x = self.conv(x)
        return torch.flatten(x, 1)

    def forward(self, x):
        return self.classifier(self.features_forward(x))


# ---------------- CIFAR-10 Vision Transformer ----------------
# A small from-scratch ViT for 32x32 CIFAR. Deliberately a *different family*
# from the two CNNs (attention-based, no conv stack beyond the patchify layer),
# so it serves as a shape-bias probe: ViTs are known to be markedly more
# shape-biased than CNNs (Naseer et al. 2021; Tuli et al. 2021). Exposes
# features_forward = the final-norm CLS token (penultimate rep) for Mahalanobis.
class _TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(h, dim), nn.Dropout(drop))

    def forward(self, x):
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, img=32, patch=4,
                 dim=192, depth=6, heads=3, mlp_ratio=2.0, drop=0.1):
        super().__init__()
        assert img % patch == 0
        n_patches = (img // patch) ** 2
        self.patch_embed = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([_TransformerBlock(dim, heads, mlp_ratio, drop)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def features_forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # B, N, dim
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]                                        # CLS token

    def forward(self, x):
        return self.head(self.features_forward(x))


ARCHS = {
    "mnist": {"lenet": LeNet, "mlp": MLP},
    "cifar10": {"resnet18": ResNet18, "vgg11": VGG11, "vit": ViT},
}


def build(dataset, arch):
    net = ARCHS[dataset][arch](num_classes=C.NUM_CLASSES[dataset], in_ch=C.IN_CHANS[dataset])
    mean, std = C.NORM[dataset]
    return NormalizedModel(net, mean, std).to(C.DEVICE)


def load(dataset, arch, tag="clean"):
    import os
    m = build(dataset, arch)
    path = os.path.join(C.CKPT_DIR, f"{dataset}_{arch}_{tag}.pt")
    m.load_state_dict(torch.load(path, map_location=C.DEVICE))
    m.eval()
    return m
