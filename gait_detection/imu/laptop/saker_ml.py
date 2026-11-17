#!/usr/bin/env python3
"""SVM gait-phase classifiers trained on recordings from saker_logger.py (scikit-learn).

    python saker_ml.py compare data/*.csv               # thigh vs shank vs both (+ rules), writes a report
    python saker_ml.py train data/*.csv --sensors both  # train one model, save models/gait_svm_<sensors>.joblib
    python saker_ml.py predict models/gait_svm_both.joblib data/x.csv   # writes data/x_pred.csv

Each sample is classified from the 200 ms before it (past only, so the model could
run live on the board). Labels come from gait_labels.py: phases found offline from
the whole recording. Pass --label phase to learn the board's rule-based phases instead.
"""
import argparse
import csv
import datetime
import pathlib
import sys

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import gait_labels

HERE = pathlib.Path(__file__).parent
FS = gait_labels.FS
PHASES = gait_labels.PHASES
SENSOR_SETS = {
    "thigh": ["thigh_sag", "thigh_gyro_mag", "thigh_acc_mag"],
    "shank": ["shank_sag", "shank_gyro_mag", "shank_acc_mag"],
    "both": ["thigh_sag", "thigh_gyro_mag", "thigh_acc_mag", "shank_sag", "shank_gyro_mag", "shank_acc_mag"],
}
MAX_TRAIN = 30000          # SVM training time grows fast with sample count
BLOCK_S = 60               # with fewer than 3 recordings, hold out 60 s blocks instead
EVENT_TOLERANCE_MS = 150   # a detected event counts if within this of the reference
DEBOUNCE = 3               # a model's phase change counts after 3 equal predictions (30 ms)


# ---------------------------------------------------------------------------
# Loading and features

def load(path):
    """One recording: numeric columns as arrays, offline labels and events added."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    cols = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            cols[key] = np.array([float(v) if v != "" else np.nan for v in values])
        except ValueError:
            cols[key] = np.array(values, dtype=object)
    labels = gait_labels.label(cols)
    # Sagittal rates from the offline axis search, so files recorded without WALK work too.
    cols["thigh_sag"], cols["shank_sag"] = labels["thigh_sag"], labels["shank_sag"]
    cols["gait"] = labels["phase"]
    for seg in ("thigh", "shank"):
        cols[f"{seg}_gyro_mag"] = np.sqrt(sum(cols[f"{seg}_g{a}"] ** 2 for a in "xyz"))
        cols[f"{seg}_acc_mag"] = np.sqrt(sum(cols[f"{seg}_a{a}"] ** 2 for a in "xyz"))
    return {"path": str(path), "cols": cols, "labels": labels, "n": len(rows)}


def windows(cols, channels, window):
    """Features for every sample with `window` clean samples behind it -> (features, sample index)."""
    n = len(cols["t_ms"])
    if n < window:
        return np.empty((0, len(channels) * 6)), np.empty(0, dtype=int)
    data = np.stack([cols[c] for c in channels], axis=1)
    w = sliding_window_view(data, window, axis=0)  # (n - window + 1, channels, window)
    ends = np.arange(window - 1, n)
    span = cols["t_ms"][ends] - cols["t_ms"][ends - window + 1]  # skip lost-packet gaps
    ok = np.isfinite(w).all(axis=(1, 2)) & (span <= (window - 1) * 1000 / FS * 1.5)
    w, ends = w[ok], ends[ok]
    feats = np.concatenate([w[:, :, -1], w.mean(2), w.std(2), w.min(2), w.max(2), w[:, :, -1] - w[:, :, 0]], axis=1)
    return feats, ends


def make_model():
    return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced"))


def label_array(rec, label):
    return np.array(["" if v is None or str(v) == "nan" else str(v) for v in rec["cols"][label]], dtype=object)


# ---------------------------------------------------------------------------
# Events from a phase sequence

def events_from_phases(idx, phases, debounce):
    """Heel strikes / toe-offs from per-sample predictions, as a live system would see them."""
    hs, to = [], []
    current, run_label, run_len, prev_i = None, None, 0, None
    for i, lab in zip(idx, phases):
        if prev_i is not None and i != prev_i + 1:  # gap in the data: start over
            current, run_label, run_len = None, None, 0
        prev_i = i
        run_label, run_len = (run_label, run_len + 1) if lab == run_label else (lab, 1)
        if run_len >= debounce and run_label != current:
            if current in ("SWING", "TERMINAL_SWING") and run_label == "STANCE":
                hs.append(i)
            if current in ("STANCE", "PRE_SWING") and run_label == "SWING":
                to.append(i)
            current = run_label
    return np.array(hs, dtype=int), np.array(to, dtype=int)


def match_events(reference, detected, regions):
    """Hits, misses, false alarms and timing errors (ms, + = late) inside labelled strides."""
    margin = int(0.2 * FS)
    inside = [d for d in detected if any(a - margin <= d <= b + margin for a, b in regions)]
    tol = EVENT_TOLERANCE_MS * FS // 1000
    used, errors = set(), []
    for r in reference:
        candidates = [d for d in inside if abs(d - r) <= tol and d not in used]
        if candidates:
            best = min(candidates, key=lambda d: abs(d - r))
            used.add(best)
            errors.append((best - r) * 1000 / FS)
    return {"reference": len(reference), "hits": len(errors), "false": len(inside) - len(used), "errors": errors}


def merge(a, b):
    return {"reference": a["reference"] + b["reference"], "hits": a["hits"] + b["hits"],
            "false": a["false"] + b["false"], "errors": a["errors"] + b["errors"]}


def event_summary(m):
    e = np.array(m["errors"]) if m["errors"] else np.array([np.nan])
    return {
        "found": m["hits"] / m["reference"] if m["reference"] else np.nan,
        "false_per_100": 100 * m["false"] / m["reference"] if m["reference"] else np.nan,
        "median_ms": float(np.median(e)),
        "p90_abs_ms": float(np.percentile(np.abs(e), 90)),
    }


# ---------------------------------------------------------------------------
# Commands

def load_all(paths):
    recs = []
    for path in paths:
        rec = load(path)
        if rec is None:
            print(f"skipping {path}: empty")
            continue
        n_lab = int((rec["cols"]["gait"] != "").sum())
        print(f"{path}: {rec['n'] / FS / 60:.1f} min, {len(rec['labels']['stride_regions'])} strides, "
              f"{n_lab / FS / 60:.1f} min labelled, axes shank {rec['labels']['axes'][0]} thigh {rec['labels']['axes'][1]}")
        recs.append(rec)
    if not recs:
        sys.exit("No usable recordings.")
    return recs


def groups_for(recs, ends_per_rec):
    """Recording index if there are at least 3 recordings, else 60 s blocks."""
    by_file = len(recs) >= 3
    groups, next_id = [], 0
    for r, ends in enumerate(ends_per_rec):
        g = np.full(len(ends), r) if by_file else next_id + ends // (BLOCK_S * FS)
        groups.append(g)
        next_id = max(next_id, int(g.max()) + 1 if len(g) else next_id)
    return groups, ("whole recordings" if by_file else f"{BLOCK_S} s blocks")


def compare(args):
    recs = load_all(args.files)
    label = args.label
    results, rows_md = {}, []
    # Same folds for every sensor set: groups depend only on sample index, not on the channels.
    _, ends0 = zip(*[windows(r["cols"], SENSOR_SETS["both"], args.window) for r in recs])
    groups, split_kind = groups_for(recs, ends0)
    n_groups = len(np.unique(np.concatenate(groups)))
    if n_groups < 2:
        sys.exit("Need at least 2 minutes of data (or 2 recordings) to hold something out.")
    folds = GroupKFold(n_splits=min(5, n_groups))
    print(f"\nholding out {split_kind}, {min(5, n_groups)} folds; labels: "
          f"{'offline shank-gyro events' if label == 'gait' else label}\n")

    for name, channels in SENSOR_SETS.items():
        X, y, g, which, ends = [], [], [], [], []
        for r, rec in enumerate(recs):
            feats, e = windows(rec["cols"], channels, args.window)
            X.append(feats); y.append(label_array(rec, label)[e]); g.append(groups[r])
            which.append(np.full(len(e), r)); ends.append(e)
        X, y, g, which, ends = map(np.concatenate, (X, y, g, which, ends))
        pred = np.full(len(y), "", dtype=object)
        n_sv = []
        rng = np.random.default_rng(0)
        for train_idx, test_idx in folds.split(X, y, g):
            train_idx = train_idx[y[train_idx] != ""]
            if len(train_idx) > MAX_TRAIN:
                train_idx = np.sort(rng.choice(train_idx, MAX_TRAIN, replace=False))
            model = make_model().fit(X[train_idx], y[train_idx])
            n_sv.append(int(model[-1].n_support_.sum()))
            pred[test_idx] = model.predict(X[test_idx])  # every window, labelled or not, for events
        has = y != ""
        hs_m = to_m = {"reference": 0, "hits": 0, "false": 0, "errors": []}
        for r, rec in enumerate(recs):
            sel = which == r
            hs, to = events_from_phases(ends[sel], pred[sel], DEBOUNCE)
            regions = rec["labels"]["stride_regions"]
            hs_m = merge(hs_m, match_events(rec["labels"]["heel_strikes"], hs, regions))
            to_m = merge(to_m, match_events(rec["labels"]["toe_offs"], to, regions))
        results[name] = {
            "accuracy": accuracy_score(y[has], pred[has]),
            "macro_f1": f1_score(y[has], pred[has], average="macro", labels=PHASES, zero_division=0),
            "per_phase": f1_score(y[has], pred[has], average=None, labels=PHASES, zero_division=0),
            "hs": event_summary(hs_m), "to": event_summary(to_m),
            "support_vectors": int(np.mean(n_sv)), "features": X.shape[1],
        }
        print(f"{name:6} done")

    # The board's own rules, if the recordings were made in TRACK / START mode.
    rule_rows = [rec for rec in recs if "phase" in rec["cols"] and (rec["cols"]["phase"] != "").any()]
    if rule_rows and label == "gait":
        ys, ps = [], []
        hs_m = to_m = {"reference": 0, "hits": 0, "false": 0, "errors": []}
        for rec in rule_rows:
            gt, ph, ev = rec["cols"]["gait"], label_array(rec, "phase"), rec["cols"]["event"]
            m = (gt != "") & (ph != "")
            ys.append(gt[m]); ps.append(ph[m])
            regions = rec["labels"]["stride_regions"]
            hs_m = merge(hs_m, match_events(rec["labels"]["heel_strikes"], np.flatnonzero(ev == "HEEL_STRIKE"), regions))
            to_m = merge(to_m, match_events(rec["labels"]["toe_offs"], np.flatnonzero(ev == "TOE_OFF"), regions))
        ys, ps = np.concatenate(ys), np.concatenate(ps)
        results["rules (board)"] = {
            "accuracy": accuracy_score(ys, ps),
            "macro_f1": f1_score(ys, ps, average="macro", labels=PHASES, zero_division=0),
            "per_phase": f1_score(ys, ps, average=None, labels=PHASES, zero_division=0),
            "hs": event_summary(hs_m), "to": event_summary(to_m), "support_vectors": 0, "features": 0,
        }

    report = render_report(recs, results, split_kind, label, args.window)
    print("\n" + report)
    out = HERE / "results" / f"compare-{datetime.datetime.now():%Y%m%d-%H%M}.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text(report)
    print(f"saved {out}")


def render_report(recs, results, split_kind, label, window):
    minutes = sum(r["n"] for r in recs) / FS / 60
    strides = sum(len(r["labels"]["stride_regions"]) for r in recs)
    pct = lambda v: "n/a" if np.isnan(v) else f"{100 * v:.0f}%"
    ms = lambda v: "n/a" if np.isnan(v) else f"{v:+.0f}"
    lines = [
        f"# SAKER sensor comparison ({datetime.datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Data: {len(recs)} recording(s), {minutes:.1f} min, {strides} labelled strides. "
        f"Tested on held-out {split_kind}; each model only sees the {window * 1000 // FS} ms before each sample.",
        f"Labels: {'offline shank-gyro gait events (gait_labels.py)' if label == 'gait' else label}.",
        "",
        "## Phases",
        "",
        "| model | macro F1 | accuracy | " + " | ".join(PHASES) + " |",
        "|---|---|---|" + "---|" * len(PHASES),
    ]
    for name, r in results.items():
        lines.append(f"| {name} | {r['macro_f1']:.3f} | {pct(r['accuracy'])} | " +
                     " | ".join(f"{v:.2f}" for v in r["per_phase"]) + " |")
    lines += [
        "",
        f"## Events (counted when within {EVENT_TOLERANCE_MS} ms; timing + = late)",
        "",
        "| model | heel strikes found | false / 100 | median timing ms | 90% within ms | "
        "toe-offs found | false / 100 | median timing ms | 90% within ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        h, t = r["hs"], r["to"]
        lines.append(f"| {name} | {pct(h['found'])} | {h['false_per_100']:.0f} | {ms(h['median_ms'])} | "
                     f"{h['p90_abs_ms']:.0f} | {pct(t['found'])} | {t['false_per_100']:.0f} | "
                     f"{ms(t['median_ms'])} | {t['p90_abs_ms']:.0f} |")
    lines += ["", "## Model size (for running on the ESP32 later)", "",
              "| model | support vectors | features | multiply-adds per prediction |", "|---|---|---|---|"]
    for name, r in results.items():
        if r["support_vectors"]:
            lines.append(f"| {name} | {r['support_vectors']} | {r['features']} | "
                         f"~{r['support_vectors'] * r['features'] / 1000:.0f}k |")
    lines += ["", "Caveat: the reference labels come from the shank gyro, so the shank-only "
              "model is graded against its own sensor and has a home advantage.", ""]
    return "\n".join(lines)


def train(args):
    recs = load_all(args.files)
    channels = SENSOR_SETS[args.sensors]
    X, y = [], []
    for rec in recs:
        feats, ends = windows(rec["cols"], channels, args.window)
        X.append(feats); y.append(label_array(rec, args.label)[ends])
    X, y = np.concatenate(X), np.concatenate(y)
    X, y = X[y != ""], y[y != ""]
    if len(y) > MAX_TRAIN:
        keep = np.sort(np.random.default_rng(0).choice(len(y), MAX_TRAIN, replace=False))
        X, y = X[keep], y[keep]
    model = make_model().fit(X, y)
    out = pathlib.Path(args.model or HERE / "models" / f"gait_svm_{args.sensors}.joblib")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "window": args.window, "label": args.label, "channels": channels}, out)
    print(f"saved {args.sensors} model ({int(model[-1].n_support_.sum())} support vectors, {len(y)} windows) to {out}")


def predict(args):
    saved = joblib.load(args.model)
    for path in args.files:
        rec = load(path)
        feats, ends = windows(rec["cols"], saved["channels"], saved["window"])
        pred = np.full(rec["n"], "", dtype=object)
        if len(ends):
            pred[ends] = saved["model"].predict(feats)
        out = pathlib.Path(path).with_name(pathlib.Path(path).stem + "_pred.csv")
        with open(path, newline="") as src, open(out, "w", newline="") as dst:
            reader, writer = csv.reader(src), csv.writer(dst)
            writer.writerow(next(reader) + ["gait_label", "pred_phase"])
            for row, g, p in zip(reader, rec["cols"]["gait"], pred):
                writer.writerow(row + [g, p])
        truth = label_array(rec, saved["label"])[ends]
        has = truth != ""
        if has.any():
            print(f"{path}: agrees with '{saved['label']}' labels on {(truth[has] == pred[ends][has]).mean():.1%}")
        print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("compare", "train"):
        s = sub.add_parser(name)
        s.add_argument("files", nargs="+")
        s.add_argument("--label", default="gait", help="'gait' = offline labels (default), 'phase' = board's rules")
        s.add_argument("--window", type=int, default=20, help="samples of history per prediction (20 = 200 ms)")
        if name == "train":
            s.add_argument("--sensors", choices=SENSOR_SETS, default="both")
            s.add_argument("--model", default=None)
    pr = sub.add_parser("predict")
    pr.add_argument("model")
    pr.add_argument("files", nargs="+")
    args = p.parse_args()
    {"compare": compare, "train": train, "predict": predict}[args.cmd](args)


if __name__ == "__main__":
    main()
