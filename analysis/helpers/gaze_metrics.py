"""
gaze_metrics.py — gaze-map metrics and letterbox cropping shared across the pipeline.

Turns fixations into a smoothed full-resolution density map (per-pixel histogram -> FFT
Gaussian low-pass at cutoff `fc` -> [0, 1]) and scores a predicted map against human gaze
with Pearson CC and NSS. Also crops the black letterbox bars the COCO-Search18 display frame
adds. Pure CPU / numpy — no model here, so it is importable and unit-testable.
"""

import numpy as np


# ----------------------------------------------------------------------------
# letterbox crop (COCO-Search18 images are padded into a fixed display frame; the
# black bars waste tokens and attract attention artifacts — crop to real content)
# ----------------------------------------------------------------------------
def detect_letterbox(image, thresh=15):
    """(left, top, right, bottom) box of the non-black content in a PIL image."""
    a = np.asarray(image.convert("L"))
    h, w = a.shape
    cols = np.where(a.max(axis=0) >= thresh)[0]
    rows = np.where(a.max(axis=1) >= thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return (0, 0, w, h)
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def crop_letterbox(image, fixations=None, thresh=15):
    """Crop black letterbox bars; shift/clip fixation arrays to match.

    fixations: dict of key -> N x 2 arrays in the padded frame (or None).
    Returns (cropped_image, cropped_fixations_or_None, (cw, ch)).
    """
    l, t, r, b = detect_letterbox(image, thresh)
    cropped = image.crop((l, t, r, b))
    cw, ch = cropped.size
    out = None
    if fixations is not None:
        out = {}
        for k, fx in fixations.items():
            if len(fx):
                g = fx.astype(np.float32).copy()
                g[:, 0] -= l; g[:, 1] -= t
                keep = (g[:, 0] >= 0) & (g[:, 0] < cw) & (g[:, 1] >= 0) & (g[:, 1] < ch)
                out[k] = g[keep]
            else:
                out[k] = fx
    return cropped, out, (cw, ch)


# ----------------------------------------------------------------------------
# maps + metrics — full-resolution fixation heatmap + Pearson CC
#
# Build a full-resolution 2-D fixation histogram (one bin per image pixel), low-pass it with
# an FFT Gaussian at cutoff frequency `fc`, then min-max to [0, 1]. FROZEN_FC = 6 gives an
# effective Gaussian sigma ~6% of image height on COCO-sized frames. The same blur is used for
# every map (human and model output) so they are comparable.
# ----------------------------------------------------------------------------
FROZEN_FC = 6.0   # low-pass cutoff frequency (cycles/image, -6 dB); one blur for every map


def _mat2gray(a):
    """Min-max normalize to [0,1] (numpy port of the repo's cv2 mat2gray)."""
    a = np.asarray(a, np.float64)
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


_GF_CACHE = {}


def _gaussian_transfer(n, fc):
    """FFT-domain Gaussian low-pass transfer function for an n x n FFT at cutoff fc, sliced to
    the rfft2 layout (n, n//2+1). Depends only on (n, fc), so it is cached across calls."""
    gf = _GF_CACHE.get((n, fc))
    if gf is None:
        fx, fy = np.meshgrid(np.arange(n), np.arange(n))
        fx = fx - n / 2.0; fy = fy - n / 2.0
        s = fc / np.sqrt(np.log(2))                      # cutoff freq -> gaussian width
        gf = np.fft.fftshift(np.exp(-(fx ** 2 + fy ** 2) / (s ** 2)))[:, :n // 2 + 1]
        _GF_CACHE[(n, fc)] = gf
    return gf


def gaussian_2d(img, fc=FROZEN_FC):
    """FFT Gaussian low-pass at cutoff frequency fc (-6 dB).

    Uses a real FFT and the cached transfer function keyed by (n, fc). This is the shared
    smoothing kernel for every density map.
    """
    sn, sm = img.shape
    n = max(sn, sm)
    n = int(2 ** np.ceil(np.log2(n + (n % 2))))          # FFT size: next power of two
    gfr = _gaussian_transfer(n, fc)
    bf = np.fft.irfft2(np.fft.rfft2(img, s=[n, n]) * gfr, s=[n, n])
    return bf[:sn, :sm]


def draw_heatmap(x_pos, y_pos, im_w, im_h, weights=None, fc=FROZEN_FC):
    """Full-resolution fixation-density map, shape (im_h, im_w), in [0,1].

    One histogram bin per pixel, FFT-Gaussian smoothed, min-max normalized. `weights`
    (per-fixation, e.g. duration or a rank weight) is optional — pass None for uniform.
    """
    hist = np.histogram2d(y_pos, x_pos, bins=(im_h, im_w),
                          range=[[0, im_h], [0, im_w]], weights=weights)[0]
    return _mat2gray(gaussian_2d(hist, fc)).astype(np.float32)


def density_map(fix_xy, im_w, im_h, fc=FROZEN_FC, weights=None):
    """Fixation/point array (N x 2 pixel x,y) -> full-resolution [0, 1] density map.

    Same smoothing scale for human and model points.
    """
    fix_xy = np.asarray(fix_xy, np.float32).reshape(-1, 2)
    if len(fix_xy) == 0:
        return np.zeros((im_h, im_w), np.float32)
    w = None if weights is None else np.asarray(weights, np.float64).reshape(-1)
    return draw_heatmap(fix_xy[:, 0], fix_xy[:, 1], im_w, im_h, weights=w, fc=fc)


def nss(pred, fix_xy, im_w, im_h):
    """Normalized Scanpath Saliency: z-score pred, average at fixation pixels.

    Resolution-agnostic — maps fixations onto pred's own grid, so it works whether pred
    is a full-resolution map or a resized model map.
    """
    if len(fix_xy) == 0:
        return np.nan
    H, W = pred.shape
    p = pred.astype(np.float64)
    s = p.std()
    if s == 0:
        return np.nan
    p = (p - p.mean()) / s
    gx = np.clip((fix_xy[:, 0] / im_w * W).astype(int), 0, W - 1)
    gy = np.clip((fix_xy[:, 1] / im_h * H).astype(int), 0, H - 1)
    return float(p[gy, gx].mean())


def cc(pred, human):
    """Pearson CC over the flattened maps."""
    a, b = np.asarray(pred, np.float64).reshape(-1), np.asarray(human, np.float64).reshape(-1)
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])
