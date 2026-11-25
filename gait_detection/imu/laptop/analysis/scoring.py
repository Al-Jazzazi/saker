"""Scoring against the manual marks, on exactly the same samples and events for
every method.

Test set of a recording = everything from setup_end (walk_normal start + 30 s)
to the end. Phase scores use received samples there whose manual label is
known. Event scores use the 'sure' manual events there; a detection counts if
it is within +-150 ms of a manual event of the same type (one-to-one).
Detections inside 'unclear' spans, or near 'unsure' events, are ignored.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

TYPES = ["HS", "PO", "TO", "MS"]
MATCH_S = 0.15


def confusion(pred, lab, mask):
    """4x4 counts, rows = manual label, columns = method, over mask & labelled samples."""
    m = mask & (lab >= 0)
    if (pred[m] < 0).any():
        raise ValueError("method has no output on some scored samples")
    cm = np.zeros((4, 4), int)
    np.add.at(cm, (lab[m], pred[m]), 1)
    return cm


def phase_scores(cm):
    tp = np.diag(cm).astype(float)
    prec = tp / np.maximum(cm.sum(0), 1)
    rec = tp / np.maximum(cm.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    present = cm.sum(1) > 0
    return dict(acc=tp.sum() / max(cm.sum(), 1), macro_f1=f1[present].mean() if present.any() else np.nan,
                f1=f1.tolist(), n=int(cm.sum()))


def merge3(cm):
    """STANCE+PRE_SWING merged: scores that don't depend on where PO is put."""
    idx = [0, 0, 1, 2]
    out = np.zeros((3, 3), int)
    for i in range(4):
        for j in range(4):
            out[idx[i], idx[j]] += cm[i, j]
    tp = np.diag(out).astype(float)
    prec = tp / np.maximum(out.sum(0), 1)
    rec = tp / np.maximum(out.sum(1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    return dict(acc=tp.sum() / max(out.sum(), 1), macro_f1=f1.mean())


def event_sets(marks, t_start, t_end):
    """Scored manual events and the spans where detections are ignored."""
    ev = [e for e in marks["events"] if t_start <= e["t"] < t_end]
    sure = {k: np.array([e["t"] for e in ev if e["type"] == k and e.get("conf") != "unsure"]) for k in TYPES}
    unsure = {k: np.array([e["t"] for e in marks["events"] if e["type"] == k and e.get("conf") == "unsure"])
              for k in TYPES}
    # true events just before t_start: detections matching them are neither hits nor false alarms
    edge = {k: np.array([e["t"] for e in marks["events"] if e["type"] == k and t_start - MATCH_S <= e["t"] < t_start])
            for k in TYPES}
    return dict(sure=sure, unsure=unsure, edge=edge, unclear=marks.get("unclear", []),
                still=marks.get("still", []), t_start=t_start, t_end=t_end)


def _match(tt, tp, window=None):
    window = MATCH_S if window is None else window
    if len(tt) == 0 or len(tp) == 0:
        return []
    cost = np.abs(tt[:, None] - tp[None, :])
    big = cost > window
    cost = np.where(big, 1e6, cost)
    r, c = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(r, c) if not big[i, j]]


def event_scores(pred_events, S, window=None):
    """pred_events: [(t, type)]. Returns per type: hits, misses, false alarms, signed errors (s).
    window: match tolerance in s (default MATCH_S = 0.15)."""
    window = MATCH_S if window is None else window
    out = {}
    for k in TYPES:
        tp = np.array(sorted(t for t, ty in pred_events if ty == k and S["t_start"] - window <= t < S["t_end"]))
        tt = S["sure"][k]
        pairs = _match(tt, tp, window)
        used = {j for _, j in pairs}
        err = np.array([tp[j] - tt[i] for i, j in pairs])
        # unmatched detections: ignore those explained by edge/unsure events or unclear spans
        rest = np.array([tp[j] for j in range(len(tp)) if j not in used])
        ignore = np.zeros(len(rest), bool)
        for ref in (S["edge"][k], S["unsure"][k]):
            if len(ref) and len(rest):
                ignore |= np.abs(rest[:, None] - ref[None, :]).min(1) <= window
        for a, b in S["unclear"]:
            ignore |= (rest >= a) & (rest < b)
        ignore |= rest < S["t_start"]
        fa = rest[~ignore]
        still_fa = sum(any(a <= t < b for a, b in S["still"]) for t in fa)
        out[k] = dict(n_true=len(tt), hits=len(pairs), misses=len(tt) - len(pairs), false_alarms=len(fa),
                      false_alarms_still=int(still_fa), err=err.tolist(), fa_times=fa.tolist())
    return out


def summarize_events(sc):
    """Pooled numbers from one or more event_scores() dicts."""
    sc = sc if isinstance(sc, list) else [sc]
    out = {}
    for k in TYPES:
        n = sum(s[k]["n_true"] for s in sc)
        h = sum(s[k]["hits"] for s in sc)
        fa = sum(s[k]["false_alarms"] for s in sc)
        err = np.concatenate([np.asarray(s[k]["err"]) for s in sc]) * 1000 if n else np.array([])
        out[k] = dict(n_true=n, recall=h / n if n else np.nan, precision=h / (h + fa) if h + fa else np.nan,
                      fa_per_100=100 * fa / n if n else np.nan, fa_still=sum(s[k]["false_alarms_still"] for s in sc),
                      err_median_ms=float(np.median(err)) if len(err) else np.nan,
                      err_mean_ms=float(np.mean(err)) if len(err) else np.nan,
                      abs_err_p50_ms=float(np.median(np.abs(err))) if len(err) else np.nan,
                      abs_err_p90_ms=float(np.percentile(np.abs(err), 90)) if len(err) else np.nan,
                      within50=float((np.abs(err) <= 50).sum() / n) if n else np.nan)
        p, r = out[k]["precision"], out[k]["recall"]
        out[k]["f1"] = 2 * p * r / (p + r) if p == p and r == r and p + r > 0 else np.nan
    return out
