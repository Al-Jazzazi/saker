"""Offline gait labels for SAKER recordings (no foot sensors needed).

Works on a whole recording at once from the raw IMU columns, so it does not
depend on the board's CAL/WALK state:
  1. Gyro bias from the quietest 2 s of the recording.
  2. Sagittal axis and sign of each IMU (same idea as the firmware's WALK):
     the shank axis that moves most, signed so mid-swing is positive; the thigh
     axis that moves most, signed to move together with the shank.
  3. Gait events with the standard shank-gyro method: mid-swing = big positive
     peak, toe-off = lowest point in the 450 ms before it, heel strike = the dip
     right after the rate falls below zero. Looking ahead like this is only
     possible offline, so these labels are a stronger reference than any live
     detector.
  4. Phases per sample, matching the firmware's names: STANCE (heel strike to
     pre-swing), PRE_SWING (last 12% of the stride before toe-off), SWING
     (toe-off to mid-swing), TERMINAL_SWING (mid-swing to heel strike).
     Standing still counts as STANCE. Anything else (turns, stumbles,
     start/stop) stays unlabelled and is left out.

Caveat: the events come from the shank sensor, so a shank-only model is graded
against its own sensor and has a home advantage over a thigh-only one.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

FS = 100  # samples per second
PHASES = ["STANCE", "PRE_SWING", "SWING", "TERMINAL_SWING"]
PRE_SWING_FRACTION = 0.12   # pre-swing ~ the last 12% of the stride before toe-off
MIN_STRIDE_S, MAX_STRIDE_S = 0.6, 2.0


def lowpass(x, hz):
    b, a = butter(2, hz / (FS / 2))
    return filtfilt(b, a, x)


def _gyro(cols, seg):
    return np.stack([cols[f"{seg}_g{a}"] for a in "xyz"], axis=1)


def _remove_bias(thigh, shank):
    """Subtract the mean of the quietest 2 s, if that stretch really was still."""
    n, w = len(thigh), 2 * FS
    if n < w:
        return thigh, shank
    both = np.concatenate([thigh, shank], axis=1)
    csum = np.cumsum(np.vstack([np.zeros(6), both]), axis=0)
    csq = np.cumsum(np.vstack([np.zeros(6), both ** 2]), axis=0)
    mean = (csum[w:] - csum[:-w]) / w
    std = np.sqrt(np.maximum(0, (csq[w:] - csq[:-w]) / w - mean ** 2))
    i = int(np.argmin(std.sum(axis=1)))
    if std[i].max() > 1.5:  # never truly still: keep the board's own bias correction
        return thigh, shank
    return thigh - mean[i, :3], shank - mean[i, 3:]


def sagittal(cols):
    """Signed sagittal rates (deg/s) and the axes chosen, e.g. ('Y-', 'Z+')."""
    thigh, shank = _remove_bias(_gyro(cols, "thigh"), _gyro(cols, "shank"))
    s_axis = int(np.argmax(shank.var(axis=0)))
    hi, lo = np.percentile(shank[:, s_axis], [98, 2])
    s_sign = 1 if hi >= -lo else -1
    shank_sag = s_sign * shank[:, s_axis]
    t_axis = int(np.argmax(thigh.var(axis=0)))
    t_sign = 1 if np.dot(thigh[:, t_axis], shank_sag) >= 0 else -1
    thigh_sag = t_sign * thigh[:, t_axis]
    names = ("XYZ"[s_axis] + "+-"[s_sign < 0], "XYZ"[t_axis] + "+-"[t_sign < 0])
    return thigh_sag, shank_sag, thigh, shank, names


def label(cols):
    """Returns dict with per-sample 'phase' labels ('' = unlabelled) and event indices."""
    thigh_sag, shank_sag, thigh, shank, axes = sagittal(cols)
    n = len(shank_sag)
    w = lowpass(shank_sag, 12)
    p98 = np.percentile(w, 98)

    # One (toe-off, mid-swing, heel-strike) triple per swing.
    peaks, _ = find_peaks(w, height=max(60.0, 0.4 * p98), distance=int(0.5 * FS))
    triples = []
    for ms in peaks:
        lo = max(0, ms - int(0.45 * FS))
        to = lo + int(np.argmin(w[lo:ms]))
        below = np.nonzero(w[ms:min(n, ms + int(0.35 * FS))] < 0)[0]
        if w[to] >= 0 or len(below) == 0:
            continue
        zc = ms + int(below[0])
        hs = zc + int(np.argmin(w[zc:min(n, zc + int(0.10 * FS))]))
        triples.append((to, ms, hs))

    phase = np.full(n, "", dtype=object)
    hs_events, to_events, regions = set(), [], []
    for (_, _, h0), (to, ms, h1) in zip(triples, triples[1:]):
        stride = (h1 - h0) / FS
        if not (h0 < to < ms < h1 and MIN_STRIDE_S <= stride <= MAX_STRIDE_S):
            continue
        pre = max(h0, to - int(round(PRE_SWING_FRACTION * (h1 - h0))))
        phase[h0:pre] = "STANCE"
        phase[pre:to] = "PRE_SWING"
        phase[to:ms] = "SWING"
        phase[ms:h1] = "TERMINAL_SWING"
        hs_events.update((h0, h1))
        to_events.append(to)
        regions.append((h0, h1))

    # Standing still (both segments quiet for at least 1 s) is stance too.
    quiet = (lowpass(np.linalg.norm(thigh, axis=1), 2) < 15) & (lowpass(np.linalg.norm(shank, axis=1), 2) < 15)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], quiet.astype(int), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= FS:
            span = phase[a:b]
            span[span == ""] = "STANCE"

    return {
        "phase": phase,
        "heel_strikes": np.array(sorted(hs_events), dtype=int),
        "toe_offs": np.array(sorted(to_events), dtype=int),
        "stride_regions": regions,
        "thigh_sag": thigh_sag,
        "shank_sag": shank_sag,
        "axes": axes,
    }
