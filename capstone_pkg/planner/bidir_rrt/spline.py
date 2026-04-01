from __future__ import annotations

from typing import List, Sequence

import numpy as np


def spline_interpolate_path(path: Sequence[Sequence[float]], dt: float = 0.01) -> List[List[float]]:
    """Interpolate joint-space waypoints with a cubic Hermite spline."""
    if dt <= 0.0:
        raise ValueError("dt must be > 0")
    if not path:
        return []

    qs = np.asarray(path, dtype=np.float64)
    if qs.ndim != 2:
        raise ValueError("path must be a 2D sequence")
    if qs.shape[0] == 1:
        return [qs[0].astype(float).tolist()]

    eps = 1e-9

    # Remove consecutive duplicates to avoid zero-length spline segments.
    diffs = np.diff(qs, axis=0)
    seg_len = np.linalg.norm(diffs, axis=1)
    keep = [0] + [i + 1 for i, d in enumerate(seg_len) if float(d) > eps]
    qs = qs[keep]
    if qs.shape[0] == 1:
        return [qs[0].astype(float).tolist()]

    # Chord-length parameterization (t in "path length" units).
    t = np.zeros(qs.shape[0], dtype=np.float64)
    t[1:] = np.cumsum(np.linalg.norm(np.diff(qs, axis=0), axis=1))
    total = float(t[-1])
    if total <= eps:
        return [qs[0].astype(float).tolist()]

    tangents = np.zeros_like(qs)
    tangents[0] = (qs[1] - qs[0]) / (t[1] - t[0])
    tangents[-1] = (qs[-1] - qs[-2]) / (t[-1] - t[-2])
    if qs.shape[0] > 2:
        denom = (t[2:] - t[:-2])[:, None]
        tangents[1:-1] = (qs[2:] - qs[:-2]) / denom

    ts = np.arange(0.0, total, dt, dtype=np.float64)
    if ts.size == 0 or abs(float(ts[-1]) - total) > eps:
        ts = np.append(ts, total)
    else:
        ts[-1] = total

    seg_idx = np.searchsorted(t, ts, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, qs.shape[0] - 2)

    t0 = t[seg_idx]
    t1 = t[seg_idx + 1]
    h = (t1 - t0)[:, None]
    u = ((ts - t0)[:, None]) / h

    q0 = qs[seg_idx]
    q1 = qs[seg_idx + 1]
    m0 = tangents[seg_idx]
    m1 = tangents[seg_idx + 1]

    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2
    out = h00 * q0 + h10 * h * m0 + h01 * q1 + h11 * h * m1

    out[0] = qs[0]
    out[-1] = qs[-1]
    return out.astype(float).tolist()

