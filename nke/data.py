"""Dataset loaders. Images are kept in [0,1] pixel space; normalization is
applied *inside* the model wrapper so that attacks and distances (eps_l) are
defined in natural pixel space and clipping to [0,1] is meaningful."""
import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np
from . import config as C


def get_dataset(name, train=False, augment=False):
    name = name.lower()
    if augment and train:
        if name == "cifar10":
            tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
        else:
            tf = T.ToTensor()
    else:
        tf = T.ToTensor()  # -> [0,1]
    if name == "mnist":
        return torchvision.datasets.MNIST(C.DATA_DIR, train=train, download=True, transform=tf)
    elif name == "cifar10":
        return torchvision.datasets.CIFAR10(C.DATA_DIR, train=train, download=True, transform=tf)
    raise ValueError(name)


def get_loader(name, train=False, batch_size=256, augment=False, shuffle=None, num_workers=4):
    ds = get_dataset(name, train=train, augment=augment)
    if shuffle is None:
        shuffle = train
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_ood_dataset(for_name):
    """A genuine-OOD reference set disjoint from the training distribution.
    For CIFAR-10 we use SVHN; for MNIST we use FashionMNIST."""
    for_name = for_name.lower()
    if for_name == "cifar10":
        return torchvision.datasets.SVHN(C.DATA_DIR, split="test", download=True,
                                         transform=T.ToTensor())
    else:
        return torchvision.datasets.FashionMNIST(C.DATA_DIR, train=False, download=True,
                                                 transform=T.ToTensor())


def sample_correct(model, name, n, batch_size=256, seed=C.SEED, stratified=True):
    """Return (x, y) of n images from the test set that `model` classifies
    correctly, optionally stratified across classes. x in [0,1]."""
    ds = get_dataset(name, train=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    model.eval()
    xs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(C.DEVICE); y = y.to(C.DEVICE)
            pred = model(x).argmax(1)
            m = pred == y
            xs.append(x[m].cpu()); ys.append(y[m].cpu())
    xs = torch.cat(xs); ys = torch.cat(ys)
    g = torch.Generator().manual_seed(seed)
    if stratified:
        ncls = C.NUM_CLASSES[name]; per = max(1, n // ncls)
        idx = []
        for c in range(ncls):
            ci = (ys == c).nonzero(as_tuple=True)[0]
            perm = ci[torch.randperm(len(ci), generator=g)][:per]
            idx.append(perm)
        idx = torch.cat(idx)
        idx = idx[torch.randperm(len(idx), generator=g)][:n]
    else:
        idx = torch.randperm(len(xs), generator=g)[:n]
    return xs[idx].clone(), ys[idx].clone()
