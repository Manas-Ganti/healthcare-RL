"""Calibration: reliability diagrams and ECE.

Calibration is the property the Brier score exists to reward, and the one SFT on winning
trajectories destroys first (it teaches the model to say 0.99 every time). Measuring it
separately from accuracy is what makes that failure visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    lower: float
    upper: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    bins: tuple[ReliabilityBin, ...]
    ece: float
    mean_confidence: float
    accuracy: float

    @property
    def overconfident(self) -> bool:
        return self.mean_confidence > self.accuracy

    def render(self) -> str:
        rows = [f"ECE={self.ece:.4f}  conf={self.mean_confidence:.4f}  acc={self.accuracy:.4f}"]
        rows += [
            f"  [{b.lower:.2f},{b.upper:.2f})  n={b.count:>5}  conf={b.mean_confidence:.3f}"
            f"  acc={b.accuracy:.3f}"
            for b in self.bins if b.count
        ]
        return "\n".join(rows)


def expected_calibration_error(
    probabilities: npt.NDArray[np.float64],
    true_indices: Sequence[int],
    n_bins: int = 10,
) -> CalibrationReport:
    """Top-label ECE over equal-width confidence bins.

    Equal-width, not equal-mass: equal-mass bins hide miscalibration in the
    high-confidence region, which is exactly the region that matters for a policy that
    has learned to assert.
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must be (n_samples, n_labels)")
    if len(true_indices) != p.shape[0]:
        raise ValueError("probabilities and true_indices differ in length")
    if not np.isfinite(p).all():
        raise ValueError("probabilities contain NaN or inf")

    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == np.asarray(true_indices)).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    ece = 0.0
    for lo, hi in pairwise(edges):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        n = int(mask.sum())
        if n:
            c, a = float(conf[mask].mean()), float(correct[mask].mean())
            ece += (n / len(conf)) * abs(c - a)
        else:
            c = a = 0.0
        bins.append(ReliabilityBin(float(lo), float(hi), n, c, a))

    return CalibrationReport(
        bins=tuple(bins),
        ece=float(ece),
        mean_confidence=float(conf.mean()),
        accuracy=float(correct.mean()),
    )


def sharpen(p: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Collapse each row to one-hot on its argmax.

    Gate B asks whether the model's own distribution Brier-scores BETTER than its argmax
    collapsed to one-hot. If it does not, calibration did not survive training.
    """
    out = np.zeros_like(p)
    out[np.arange(p.shape[0]), p.argmax(axis=1)] = 1.0
    return out
