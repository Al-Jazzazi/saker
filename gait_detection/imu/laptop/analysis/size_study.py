"""Model size and input size vs accuracy, on the same test set as benchmark.py
(leave-one-recording-out, scored after walk_normal start + 30 s).

  input size : how much past signal the features may use (5..100 samples = 50 ms..1 s)
  model size : MLP hidden units, random-forest trees/depth, boosting rounds/leaves
For each: phase macro F1, heel-strike F1 and false alarms, plus the model's memory
(float32 / compact tree nodes) and work per prediction, to judge what fits an ESP32
(~520 KB RAM, a few MB of flash).
Usage: size_study.py  -> results/benchmark/size_study.csv
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import methods as M  # noqa: E402
import scoring as S  # noqa: E402

HIST = (5, 10, 20, 50, 100)          # samples of history the features may use (10 ms each)
DEBOUNCE_K = 2                        # the most common choice in benchmark.py
NODE_BYTES = 12                       # feature id + threshold + child index, packed


def features_hist(R, h):
    """Same feature recipe as methods.features, limited to the last h samples."""
    cols = []
    for key in ("shank_sag", "thigh_sag", "shank_acc_dyn", "thigh_mag"):
        x = R.sig[key]
        cols.append(x)
        cols += [x - M._lag(x, k) for k in M.LAGS if k < h]
        w = min(M.W_SHORT, h)
        if w >= 2:
            win = sliding_window_view(np.concatenate([np.full(w - 1, x[0]), x]), w)
            cols += [win.min(1), win.max(1), win.mean(1), win.std(1),
                     (w - 1 - win.argmax(1)).astype(float), (w - 1 - win.argmin(1)).astype(float)]
        if key in ("shank_sag", "thigh_sag"):
            for lw in M.W_LONG:
                if lw <= h:
                    v = sliding_window_view(np.concatenate([np.full(lw - 1, x[0]), x]), lw)
                    cols += [v.min(1), v.max(1), (lw - 1 - v.argmax(1)).astype(float),
                             (lw - 1 - v.argmin(1)).astype(float)]
    return np.stack(cols, 1).astype(np.float32)


def size_of(model):
    """(memory bytes, operations per prediction, description)."""
    if isinstance(model, M.MLP):
        return 4 * model.n_params(), model.macs(), f"{model.n_params()} weights"
    if isinstance(model, RandomForestClassifier):
        nodes = sum(e.tree_.node_count for e in model.estimators_)
        depth = sum(e.tree_.max_depth for e in model.estimators_)
        return NODE_BYTES * nodes, depth, f"{len(model.estimators_)} trees, {nodes} nodes"
    if isinstance(model, HistGradientBoostingClassifier):
        trees = [p for it in model._predictors for p in it]
        nodes = sum(len(p.nodes) for p in trees)
        depth = sum(int(p.nodes["depth"].max()) for p in trees)
        return NODE_BYTES * nodes, depth, f"{len(trees)} trees, {nodes} nodes"
    raise TypeError(model)


def configs():
    out = []
    for h in HIST:   # input size, with the benchmark's default models
        out += [("input", "MLP 32-16", h, ("mlp", (32, 16))), ("input", "Gradient boosting", h, ("hgb", 200, 31))]
    for hid in ((8,), (16,), (16, 8), (32, 16), (64, 32), (128, 64)):
        out.append(("model", "MLP " + "-".join(map(str, hid)), 100, ("mlp", hid)))
    for trees, depth in ((10, 8), (25, 10), (50, 12)):
        out.append(("model", f"Random forest {trees} trees, depth {depth or 'free'}", 100, ("rf", trees, depth)))
    for it, leaves in ((25, 15), (50, 15), (100, 31), (200, 31)):
        out.append(("model", f"Gradient boosting {it} rounds, {leaves} leaves", 100, ("hgb", it, leaves)))
    return out


def fit(spec, X, y):
    if spec[0] == "mlp":
        return M.MLP(X.shape[1], spec[1]).fit(X, y, epochs=B.EPOCHS, class_weight=M.balanced_weights(y))
    if spec[0] == "rf":
        return RandomForestClassifier(spec[1], max_depth=spec[2], min_samples_leaf=5, class_weight="balanced_subsample",
                                      n_jobs=1, random_state=0).fit(X[::3], y[::3])
    return HistGradientBoostingClassifier(max_iter=spec[1], max_leaf_nodes=spec[2], class_weight="balanced",
                                          random_state=0).fit(X, y)


def run(cfg):
    kind, name, h, spec = cfg
    R = {r: B.prep(r) for r in B.RECS}
    X = {r: features_hist(R[r], h) for r in B.RECS}
    cms, evs, size = [], [], None
    for test in B.RECS:
        train = [r for r in B.RECS if r != test]
        rows = [B.train_rows(R[r]) for r in train]
        m = fit(spec, np.concatenate([X[r][k] for r, k in zip(train, rows)]),
                np.concatenate([R[r].lab[k] for r, k in zip(train, rows)]))
        T = R[test]
        ph = M.debounce(m.predict(X[test]).astype(int), DEBOUNCE_K)
        g = M.cycle_gate(ph, T.tg)
        cms.append(S.confusion(ph, T.lab, T.test_mask))
        evs.append((S.event_scores(M.events_from_phases(ph, T.tg), T.ev),
                    S.event_scores(M.events_from_phases(g, T.tg), T.ev)))
        size = size_of(m)
    p = S.phase_scores(np.sum(cms, 0))
    e = S.summarize_events([a for a, _ in evs])
    eg = S.summarize_events([b for _, b in evs])
    return dict(study=kind, model=name, history_ms=10 * h, n_features=X[B.RECS[0]].shape[1],
                macro_f1=round(p["macro_f1"], 4), HS_recall=round(e["HS"]["recall"], 4),
                HS_false_per_100=round(e["HS"]["fa_per_100"], 1), HS_f1=round(e["HS"]["f1"], 4),
                HS_false_per_100_gated=round(eg["HS"]["fa_per_100"], 1),
                memory_kB=round(size[0] / 1024, 1), ops_per_prediction=size[1], size=size[2],
                input_buffer_bytes=4 * 4 * h)


if __name__ == "__main__":
    rows = []
    import os
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        for r in ex.map(run, configs()):
            rows.append(r)
            print(r["study"], r["model"], r["history_ms"], "ms", r["macro_f1"], r["memory_kB"], "kB", flush=True)
    out = B.OUT / "size_study.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)
