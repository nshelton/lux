"""Depth-map comparison metrics.

All metrics are computed only over pixels that are valid in *both* the
prediction and the ground truth, except :data:`completeness`, which explicitly
measures how much of the ground-truth surface a method recovered. Reporting
both an error metric and completeness matters: a method can look very accurate
simply by abstaining on every hard pixel.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class DepthMetrics:
    rmse_mm: float          # root-mean-square depth error over shared-valid pixels
    mae_mm: float           # mean absolute error
    median_ae_mm: float     # median absolute error (robust to outliers)
    bad_pixel_pct: float    # % of shared-valid pixels with |err| > threshold
    completeness_pct: float # % of GT-valid pixels the method also predicted
    n_valid: int            # number of shared-valid pixels scored

    def as_dict(self) -> dict:
        return asdict(self)


def compare_depth(
    pred: np.ndarray,
    gt: np.ndarray,
    gt_mask: np.ndarray | None = None,
    bad_threshold_mm: float = 2.0,
) -> DepthMetrics:
    """Score a predicted depth map (metres) against ground truth (metres).

    Errors are reported in millimetres for readability.
    """
    if gt_mask is None:
        gt_mask = np.isfinite(gt) & (gt > 0)

    pred_valid = np.isfinite(pred) & (pred > 0)
    both = gt_mask & pred_valid

    n_gt = int(gt_mask.sum())
    n_both = int(both.sum())
    completeness = 100.0 * n_both / max(n_gt, 1)

    if n_both == 0:
        return DepthMetrics(np.nan, np.nan, np.nan, np.nan, completeness, 0)

    err_mm = (pred[both] - gt[both]) * 1000.0
    abs_err = np.abs(err_mm)
    return DepthMetrics(
        rmse_mm=float(np.sqrt(np.mean(err_mm**2))),
        mae_mm=float(np.mean(abs_err)),
        median_ae_mm=float(np.median(abs_err)),
        bad_pixel_pct=float(100.0 * np.mean(abs_err > bad_threshold_mm)),
        completeness_pct=float(completeness),
        n_valid=n_both,
    )
