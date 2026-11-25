"""Interpretability: what the ML models rely on, and where every method goes wrong.

1. Permutation importance (leave-one-recording-out): shuffle one group of
   features in the test recording and measure how much phase macro F1 and
   heel-strike F1 drop. Groups by signal (shank rate, thigh rate, shank
   acceleration, thigh rotation speed) and by kind (value now, recent change,
   200 ms window, 0.5-1 s window).
2. The 15 single features the random forest leans on most (impurity importance).
3. Where errors sit: share of each method's wrong samples that lie within
   100 ms of a true phase change (late/early boundaries) vs mid-phase.
Usage: interpret.py  -> results/benchmark/interpret_*.csv
"""
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark as B  # noqa: E402
import methods as M  # noqa: E402
import scoring as S  # noqa: E402

K = 2
SIGNALS = {"shank rate": "shank_sag", "thigh rate": "thigh_sag", "shank acceleration": "shank_acc_dyn",
           "thigh rotation speed": "thigh_mag"}


def kind_of(name):
    if any(name.endswith(f"{s}{w}") for w in M.W_LONG for s in ("min", "max", "since_max", "since_min")):
        return "0.5-1 s window"
    if any(name.endswith(f"_{s}{M.W_SHORT}") for s in ("min", "max", "mean", "std", "since_max", "since_min")):
        return "200 ms window"
    if "_d" in name.replace("_dyn", ""):
        return "recent change"
    return "value now"


def groups(names):
    g = {f"signal: {k}": [i for i, n in enumerate(names) if n.startswith(v)] for k, v in SIGNALS.items()}
    for kind in ("value now", "recent change", "200 ms window", "0.5-1 s window"):
        g[f"kind: {kind}"] = [i for i, n in enumerate(names) if kind_of(n) == kind]
    return g


def score(model, X, T):
    ph = M.debounce(model.predict(X).astype(int), K)
    f1 = S.phase_scores(S.confusion(ph, T.lab, T.test_mask))["macro_f1"]
    hs = S.summarize_events(S.event_scores(M.events_from_phases(ph, T.tg), T.ev))["HS"]["f1"]
    return f1, hs, ph


def fold(test):
    rng = np.random.default_rng(0)
    R = {r: B.prep(r) for r in B.RECS}
    train = [r for r in B.RECS if r != test]
    X = np.concatenate([R[r].X[B.train_rows(R[r])] for r in train])
    y = np.concatenate([R[r].lab[B.train_rows(R[r])] for r in train])
    T = R[test]
    names = T.names
    models = {"Random forest": B.ZOO["rf"]().fit(X[::3], y[::3]), "Gradient boosting": B.ZOO["hgb"]().fit(X, y),
              "Neural net (MLP)": M.MLP(X.shape[1], (32, 16)).fit(X, y, epochs=B.EPOCHS,
                                                                class_weight=M.balanced_weights(y))}
    rows, rf_imp = [], models["Random forest"].feature_importances_
    for mname, m in models.items():
        f0, h0, _ = score(m, T.X, T)
        for gname, cols in groups(names).items():
            Xp = T.X.copy()
            Xp[:, cols] = Xp[rng.permutation(len(Xp))][:, cols]
            f1, h1, _ = score(m, Xp, T)
            rows.append(dict(test=test, model=mname, group=gname, n_features=len(cols),
                             macro_f1_drop=f0 - f1, hs_f1_drop=np.nan_to_num(h0) - np.nan_to_num(h1)))
    return rows, rf_imp, names


def boundary_errors(raw, Ls):
    """Share of wrong samples within 100 ms of a true phase change."""
    out = []
    for mkey in ("fsm", "ml@loro", "rf@loro", "hgb@loro"):
        wrong = near = 0
        for r, L in Ls.items():
            run = raw["fsm"][r] if mkey == "fsm" else next(f for f in raw["folds"] if f["scheme"] == "loro"
                                                          and f["test"] == r)["runs"][mkey.split("@")[0]]
            ph = run["phase"].astype(int)
            m = L.test_mask & (L.lab >= 0)
            ch = np.flatnonzero(np.diff(L.lab) != 0)
            dist = np.full(len(L.lab), 1e9)
            if len(ch):
                idx = np.searchsorted(ch, np.arange(len(L.lab)))
                for c in (np.clip(idx - 1, 0, len(ch) - 1), np.clip(idx, 0, len(ch) - 1)):
                    dist = np.minimum(dist, np.abs(np.arange(len(L.lab)) - ch[c]))
            bad = m & (ph != L.lab)
            wrong += bad.sum()
            near += (bad & (dist <= 10)).sum()
        out.append(dict(method=mkey, wrong_samples=int(wrong), share_within_100ms_of_phase_change=round(near / wrong, 4)))
    return out


def errors_only():
    import analyze as A
    raw = np.load(B.OUT / "raw_results.npz", allow_pickle=True)["results"][0]
    be = boundary_errors(raw, {r: A.Lite(r) for r in B.RECS})
    with open(B.OUT / "interpret_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(be[0]))
        w.writeheader()
        w.writerows(be)
    print("wrote interpret_errors.csv")


if __name__ == "__main__" and sys.argv[1:] == ["errors"]:
    errors_only()
elif __name__ == "__main__":
    import os
    allrows, imps, names = [], [], None
    with ProcessPoolExecutor(int(os.environ.get("WORKERS", 6))) as ex:
        for rows, imp, nm in ex.map(fold, B.RECS):
            allrows += rows
            imps.append(imp)
            names = nm
            print("fold", rows[0]["test"], flush=True)
    summ = {}
    for r in allrows:
        k = (r["model"], r["group"])
        summ.setdefault(k, []).append((r["macro_f1_drop"], r["hs_f1_drop"], r["n_features"]))
    out = [dict(model=m, group=g, n_features=v[0][2], macro_f1_drop=round(float(np.mean([a for a, _, _ in v])), 4),
                macro_f1_drop_min=round(float(np.min([a for a, _, _ in v])), 4),
                macro_f1_drop_max=round(float(np.max([a for a, _, _ in v])), 4),
                hs_f1_drop=round(float(np.mean([b for _, b, _ in v])), 4)) for (m, g), v in summ.items()]
    with open(B.OUT / "interpret_importance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    imp = np.mean(imps, 0)
    top = [dict(rank=i + 1, feature=names[j], kind=kind_of(names[j]), importance=round(float(imp[j]), 4))
           for i, j in enumerate(np.argsort(-imp)[:15])]
    with open(B.OUT / "interpret_top_features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0]))
        w.writeheader()
        w.writerows(top)
    print("wrote interpret_importance.csv, interpret_top_features.csv")
