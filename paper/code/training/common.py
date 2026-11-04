"""
Common utilities for the four-stage grokking / LAR / free-energy experiments.

Metric definitions (see Shehper & Vaswani, arXiv:2605.28975):
  LAR  = log_n( ||W X||_RMS / (||W||_RMS ||X||_RMS) )
       = 1 + 1/2 * log_n( sum_i p_i q_i )
  W in R^{m x n} = unembedding matrix, X in R^{b x n} = activation batch.
  p_i = s_i^2 / sum_j s_j^2          (normalized squared singular values of W)
  q_i = sum_x (v_i^T x)^2 / sum_x ||x||^2  (activation energy on v_i)
  k_eff = n ** (2 * (1 - LAR))       (LAR-predicted effective dimension)
  Spectral entropy: H(p) = -sum p_i ln p_i / ln r, normalized to [0, 1].
"""
import os
import math
import random
import json

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ----------------------------------------------------------------------------
# Spectral / alignment metrics (computed on CPU float64 for stability)
# ----------------------------------------------------------------------------

def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def spectral_entropy(eig_or_sv):
    """Normalized Shannon entropy of a non-negative spectrum (singular values
    or eigenvalues).  H = -sum p ln p / ln(r)  in [0, 1]."""
    e = np.abs(np.asarray(eig_or_sv, dtype=np.float64))
    p = e ** 2
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    p = p[p > 0]
    r = len(e)
    if r <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(r))


def shannon_entropy_hist(weights, bins=100, range=None):
    """Shannon entropy (normalized to [0,1]) of the histogram of weight values."""
    w = np.asarray(weights, dtype=np.float64).ravel()
    hist, _ = np.histogram(w, bins=bins, range=range)
    p = hist.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(bins))


@torch.no_grad()
def lar_and_spectra(W, X):
    """
    W : (m, n) weight matrix (unembedding).
    X : (b, n) batch of activations feeding W.
    Returns dict with LAR, weight spectral entropy Sw, activation spectral
    entropy Sa, LAR-predicted k_eff, PCA-95 dimension on centered activations.
    """
    Wn = _to_np(W)
    Xn = _to_np(X)
    m, n = Wn.shape

    # ---- LAR via RMS norms (Eq. 1) ----
    W_rms = math.sqrt((Wn ** 2).mean())
    X_rms = math.sqrt((Xn ** 2).mean())
    WX = Xn @ Wn.T
    WX_rms = math.sqrt((WX ** 2).mean())
    if W_rms * X_rms > 0:
        lar = math.log(WX_rms / (W_rms * X_rms)) / math.log(n)
    else:
        lar = 0.0

    # ---- SVD of W: p distribution ----
    sv = np.linalg.svd(Wn, compute_uv=False)
    sw = spectral_entropy(sv)

    # ---- activation spectra ----
    # (a) q on right-singular vectors (uncentered, matches LAR theory)
    Vt = np.linalg.svd(Wn, compute_uv=True)[2]          # (r', n)
    proj = (Xn @ Vt.T) ** 2                              # (b, r')
    q = proj.sum(axis=0) / max((Xn ** 2).sum(), 1e-300)
    sa = spectral_entropy(np.sqrt(q))                    # spectrum -> singular-like
    # (b) PCA 95% on centered activations
    Xc = Xn - Xn.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / max(Xc.shape[0] - 1, 1)
    ev = np.linalg.eigvalsh(cov)[::-1]
    ev = np.clip(ev, a_min=0.0, a_max=None)
    csum = np.cumsum(ev) / max(ev.sum(), 1e-300)
    pca95 = int(np.searchsorted(csum, 0.95) + 1)

    k_eff = float(n ** (2.0 * (1.0 - lar)))
    return {
        "lar": float(lar),
        "k_eff": k_eff,
        "pca95": pca95,
        "Sw": float(sw),
        "Sa": float(sa),
    }


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------

def save_fig(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, name + "." + ext),
                    bbox_inches="tight", dpi=200)


def seed_curves(ax, df, value, steps=None, color=None, label=None, alpha_fill=0.2):
    """Plot mean +/- std of `value` across seeds (wide form: columns seed/step/value)."""
    import pandas as pd
    pivot = df.pivot_table(index="step", columns="seed", values=value)
    pivot = pivot.dropna(how="all")
    if steps is None:
        steps = pivot.index.to_numpy()
    else:
        steps = [s for s in steps if s in pivot.index]
    pivot = pivot.loc[steps]
    mean = pivot.mean(axis=1).to_numpy()
    std = pivot.std(axis=1).to_numpy()
    ax.plot(steps, mean, color=color, label=label, lw=1.8)
    ax.fill_between(steps, mean - std, mean + std, color=color,
                    alpha=alpha_fill, lw=0)
    return mean, std


def save_config(out_dir, cfg):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
