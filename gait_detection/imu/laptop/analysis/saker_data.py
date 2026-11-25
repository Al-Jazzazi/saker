"""Shared loading for the manual-annotation benchmark (FSM vs ML vs ML + online).

Every method sees the same signals, built the way the board builds them:
  - gyro bias = mean of the first 300 samples (3 s) of the opening stand,
    like the board's still calibration;
  - shank sagittal axis and sign = the board's WALK result on the first 12 s
    of walk_normal (laptop/results/walk_cal_thresholds.csv);
  - thigh sagittal = the thigh gyro projected on its main rotation direction
    in that same 12 s (the thigh board sits rotated, so no single axis fits).
Time is the board clock (t_ms), in seconds from the first sample. Wi-Fi gaps
are left as gaps here; `grid()` fills them on a regular 100 Hz grid for the
methods that need a continuous stream (the board itself never sees a gap).
"""
import csv
from pathlib import Path

import numpy as np

LAPTOP = Path(__file__).resolve().parents[1]
DATA = LAPTOP / "data"
ANNOT = LAPTOP / "annotations"
FS = 100
CAL_S = 12.0      # FSM walk calibration: first 12 s of walk_normal
SETUP_S = 30.0    # first 30 s of walk_normal (and the opening stand) are never scored

RECORDINGS = {
    "YR1": "20261117-173946_yousef_right-leg_sneakers_take1-partial",
    "YR2": "20261117-174833_yousef_right-leg_sneakers_take2",
    "AL1": "20261124-210920_amjad_left-leg_slippers_take1",
    "AL2": "20261124-212205_amjad_left-leg_slippers_take2",
    "YL1": "20261124-213443_yousef_left-leg_sneakers_take1-partial",
    "YL2": "20261124-213924_yousef_left-leg_sneakers_take2",
    "YL3": "20261124-214729_yousef_left-leg_sneakers_take3",
}
PERSON = {k: "amjad" if k.startswith("A") else "yousef" for k in RECORDINGS}
AXES = "XYZ"


def _walk_cal_axis(name):
    """Shank (axis index, sign) from the board's WALK calibration of walk_normal."""
    with open(LAPTOP / "results" / "walk_cal_thresholds.csv") as f:
        for row in csv.DictReader(f):
            if row["recording"] == name + ".csv" and row["part"] == "walk_normal":
                a = row["shank_axis"]
                return AXES.index(a[0]), (1 if a[1] == "+" else -1), row
    raise KeyError(name)


def load(rec):
    """One recording as a dict of arrays over the received samples."""
    name = RECORDINGS[rec]
    with open(DATA / (name + ".csv")) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])
    t_ms = col("t_ms")
    t = (t_ms - t_ms[0]) / 1000.0
    task = np.array([r["task"] for r in rows])
    thigh = np.stack([col(f"thigh_g{a}") for a in "xyz"], axis=1)
    shank = np.stack([col(f"shank_g{a}") for a in "xyz"], axis=1)
    shank_acc = np.stack([col(f"shank_a{a}") for a in "xyz"], axis=1)
    thigh_acc = np.stack([col(f"thigh_a{a}") for a in "xyz"], axis=1)

    stand = np.flatnonzero(task == "stand")[:300]
    thigh_bias, shank_bias = thigh[stand].mean(0), shank[stand].mean(0)
    thigh_c, shank_c = thigh - thigh_bias, shank - shank_bias

    walk0 = t[np.flatnonzero(task == "walk_normal")[0]]
    cal = (t >= walk0) & (t < walk0 + CAL_S)
    s_axis, s_sign, cal_row = _walk_cal_axis(name)
    shank_sag = s_sign * shank_c[:, s_axis]
    # Thigh: main rotation direction of the thigh gyro during calibration walking.
    _, _, vt = np.linalg.svd(thigh_c[cal] - thigh_c[cal].mean(0), full_matrices=False)
    u = vt[0] if np.dot(thigh_c[cal] @ vt[0], shank_sag[cal]) >= 0 else -vt[0]
    thigh_sag = thigh_c @ u

    gi = np.rint(t * FS).astype(int)          # index on the regular 100 Hz grid
    keep = np.concatenate([[True], np.diff(gi) > 0])   # drop the rare clock-jitter duplicate
    out = dict(
        rec=rec, name=name, person=PERSON[rec], t=t, gi=gi, task=task,
        shank_sag=shank_sag, thigh_sag=thigh_sag,
        thigh_mag=np.linalg.norm(thigh_c, axis=1), shank_mag=np.linalg.norm(shank_c, axis=1),
        shank_c=shank_c, thigh_c=thigh_c, shank_acc=shank_acc, thigh_acc=thigh_acc,
        shank_acc_dyn=np.linalg.norm(shank_acc, axis=1) - 1.0,
        walk0=walk0, cal_end=walk0 + CAL_S, setup_end=walk0 + SETUP_S,
        shank_axis=AXES[s_axis] + ("+" if s_sign > 0 else "-"), thigh_dir=u,
        fsm_thresholds=(float(cal_row["swingEnter"]), float(cal_row["pushOffEnter"]),
                        float(cal_row["heelStrike"])),
    )
    for k, v in list(out.items()):
        if isinstance(v, np.ndarray) and len(v) == len(t):
            out[k] = v[keep]
    return out


def gaps(d, min_missing=1):
    """(start, end) seconds of every Wi-Fi gap: the time between two received samples."""
    gi = d["gi"]
    j = np.flatnonzero(np.diff(gi) > min_missing)
    return [(gi[i] / FS, gi[i + 1] / FS) for i in j]


def grid(d, keys=("shank_sag", "thigh_sag", "thigh_mag", "shank_acc_dyn")):
    """Signals on the full 100 Hz grid, gaps filled by straight lines.
    Returns (t_grid, received_mask, {key: array}, task_grid)."""
    gi = d["gi"]
    n = gi[-1] + 1
    tg = np.arange(n) / FS
    received = np.zeros(n, bool)
    received[gi] = True
    sig = {k: np.interp(tg, gi / FS, d[k]) for k in keys}
    idx = np.searchsorted(gi, np.arange(n), side="right") - 1
    return tg, received, sig, d["task"][idx]


def task_segments(d):
    """[(task, start_s, end_s)] in recording order."""
    t, task = d["t"], d["task"]
    cuts = np.flatnonzero(task[1:] != task[:-1]) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(t)]])
    return [(task[a], t[a], t[b - 1]) for a, b in zip(starts, ends)]
