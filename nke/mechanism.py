"""Shape-vs-texture mechanistic metrics (Sec 3.4):
  - edge-preservation: agreement of Canny edge maps (clean vs perturbed)
  - texture-similarity: correlation of GLCM texture features + Gram-matrix cosine
"""
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops


def _to_uint8_gray(img):
    """img: HxW or HxWxC float in [0,1] -> uint8 grayscale HxW."""
    a = np.asarray(img)
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        a = (a.squeeze() * 255).astype(np.uint8)
    return a


def edge_preservation(clean, pert, lo=50, hi=150):
    """IoU / F1 of Canny edge pixels between clean and perturbed images."""
    ec = cv2.Canny(_to_uint8_gray(clean), lo, hi) > 0
    ep = cv2.Canny(_to_uint8_gray(pert), lo, hi) > 0
    inter = np.logical_and(ec, ep).sum()
    tp = inter
    fp = np.logical_and(ep, ~ec).sum()
    fn = np.logical_and(~ep, ec).sum()
    f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    iou = inter / (np.logical_or(ec, ep).sum() + 1e-9)
    return float(f1), float(iou)


def glcm_features(gray):
    g = (gray // 32).astype(np.uint8)  # 8 levels
    glcm = graycomatrix(g, distances=[1], angles=[0, np.pi / 2], levels=8,
                        symmetric=True, normed=True)
    props = ["contrast", "homogeneity", "energy", "correlation"]
    return np.array([graycoprops(glcm, p).mean() for p in props])


def texture_similarity(clean, pert):
    """1 - normalized L2 distance between GLCM texture-feature vectors (in [~0,1])."""
    fc = glcm_features(_to_uint8_gray(clean))
    fp = glcm_features(_to_uint8_gray(pert))
    d = np.linalg.norm(fc - fp) / (np.linalg.norm(fc) + 1e-9)
    return float(np.clip(1 - d, 0, 1))


def gram_cosine(clean, pert):
    """Cosine similarity of channel-wise Gram matrices (low-level texture)."""
    def gram(img):
        a = np.asarray(img)
        if a.ndim == 2:
            a = a[..., None]
        H, W, Cc = a.shape
        f = a.reshape(-1, Cc)
        G = f.T @ f
        return G.flatten()
    gc, gp = gram(clean), gram(pert)
    return float(np.dot(gc, gp) / (np.linalg.norm(gc) * np.linalg.norm(gp) + 1e-9))
