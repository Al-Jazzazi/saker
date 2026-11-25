"""Three-way comparison on the manual labels: board FSM vs ML vs ML + online learning.

Same test set for every method: in each recording, everything from
setup_end = walk_normal start + 30 s to the end (see scoring.py). The 30 s setup
window is where the FSM calibrates (first 12 s) and where ML + online learning
fine-tunes (first x s, x <= 30); plain ML never looks at it.

ML is trained on other recordings only:
  loro  leave one recording out (train on the other 6)
  lopo  leave one person out (Amjad's takes tested by a model trained on Yousef's, and back)
Hyper-parameters are picked on one inner validation recording, never on the test one.
Usage: benchmark.py            -> results/benchmark/
"""
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb  # noqa: E402
import methods as M  # noqa: E402
import saker_data as sd  # noqa: E402
import scoring as S  # noqa: E402

RECS = list(sd.RECORDINGS)
X_LIST = (6, 12, 20, 30)
SOURCES = ("manual", "fsm", "lookahead")
CONFIGS = [dict(hidden=(32, 16), balanced=True), dict(hidden=(64, 32), balanced=True),
           dict(hidden=(32, 16), balanced=False)]
DEBOUNCE = (1, 2, 3, 5)
FT_PROTOCOLS = [dict(lr=1e-3, epochs=10, anchor_l2=0.0), dict(lr=3e-4, epochs=20, anchor_l2=0.0),
                dict(lr=1e-3, epochs=20, anchor_l2=1e-2)]
EPOCHS = 25
# Other model families on the same features (no online learning: only the MLP is fine-tuned).
ZOO = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
    "linsvm": lambda: make_pipeline(StandardScaler(), LinearSVC(C=0.1, class_weight="balanced", max_iter=3000)),
    "rf": lambda: RandomForestClassifier(150, min_samples_leaf=5, class_weight="balanced_subsample",
                                         n_jobs=1, random_state=0),
    "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, class_weight="balanced", random_state=0),
}
ZOO_STRIDE = {"rf": 3}   # rows used for training (100 Hz rows are highly correlated)
SENSORS = {"shank": ("shank_sag", "shank_acc_dyn"), "thigh": ("thigh_sag", "thigh_mag")}
OUT = sd.LAPTOP / "results" / "benchmark"


# ---------------------------------------------------------------- data

def prep(rec):
    R = M.Rec(rec)
    R.marks = lb.load_marks(rec)
    R.lab = lb.phase_labels(R.marks, R.tg)
    R.X, R.names = M.features(R)
    R.test_mask = (R.tg >= R.setup_end) & R.received
    R.ev = S.event_sets(R.marks, R.setup_end, R.tg[-1] + 0.01)
    return R


def train_rows(R):
    return R.received & (R.lab >= 0)


# ---------------------------------------------------------------- pseudo-labels for online learning

def lookahead_labels(R, a, b):
    """Offline shank-gyro labeller (Salarian/Aminian style, like gait_labels.py) run on the
    buffered window only: something the board could do on its own buffer."""
    s = R.sig["shank_sag"][a:b]
    lab = np.full(b - a, -1, int)
    if len(s) < 50:
        return lab
    bb, aa = butter(2, 12 / (sd.FS / 2))
    w = filtfilt(bb, aa, s)
    peaks, _ = find_peaks(w, height=max(60.0, 0.4 * np.percentile(w, 98)), distance=int(0.5 * sd.FS))
    trip = []
    for ms in peaks:
        lo = max(0, ms - 45)
        to = lo + int(np.argmin(w[lo:ms])) if ms > lo else ms
        below = np.flatnonzero(w[ms:ms + 35] < 0)
        if w[to] >= 0 or not len(below):
            continue
        zc = ms + int(below[0])
        hs = zc + int(np.argmin(w[zc:zc + 10]))
        trip.append((to, ms, hs))
    for (_, _, h0), (to, ms, h1) in zip(trip, trip[1:]):
        if not (h0 < to < ms < h1 and 60 <= h1 - h0 <= 200):
            continue
        pre = max(h0, to - int(round(0.12 * (h1 - h0))))
        lab[h0:pre], lab[pre:to], lab[to:ms], lab[ms:h1] = 0, 1, 2, 3
    return lab


def finetune_labels(R, x, source):
    a, b = R.idx(R.walk0), R.idx(R.walk0 + x)
    if source == "manual":
        lab = R.lab[a:b].copy()
    elif source == "fsm":
        thr = M.walk_cal(R, min(x, sd.CAL_S))
        phase, _ = M.run_fsm(R, thr, R.walk0, R.walk0 + x)
        lab = phase[a:b].copy()
    else:
        lab = lookahead_labels(R, a, b)
    keep = (lab >= 0) & R.received[a:b]
    return np.arange(a, b)[keep], lab[keep]


# ---------------------------------------------------------------- ML pieces

def fit_model(Xs, ys, cfg, seed=0):
    X, y = np.concatenate(Xs), np.concatenate(ys)
    m = M.MLP(X.shape[1], cfg["hidden"], seed=seed)
    return m.fit(X, y, epochs=EPOCHS, lr=1e-3, class_weight=M.balanced_weights(y) if cfg["balanced"] else None,
                 seed=seed)


def ml_outputs(model, R, k):
    raw = model.predict(R.X)
    ph = M.debounce(raw, k)
    return ph, M.events_from_phases(ph, R.tg)


def quick_score(ph, ev, R):
    cm = S.confusion(ph, R.lab, R.test_mask)
    es = S.summarize_events(S.event_scores(ev, R.ev))
    return S.phase_scores(cm)["macro_f1"], es["HS"]["f1"], es


def finetune(model, R, x, source, proto, seed=0, cols=None):
    rows, y = finetune_labels(R, x, source)
    m = model.copy()
    if len(rows) < 50:
        return m
    X = R.X[rows] if cols is None else R.X[rows][:, cols]
    return m.fit(X, y, epochs=proto["epochs"], lr=proto["lr"], batch=64, l2=1e-4,
                 anchor=model if proto["anchor_l2"] else None, anchor_l2=proto["anchor_l2"],
                 set_scaler=False, seed=seed)


def run_fold(job):
    """One test recording: select on an inner validation recording, train, test,
    then fine-tune for every (source, x)."""
    scheme, test, train, val = job["scheme"], job["test"], job["train"], job["val"]
    R = {r: prep(r) for r in set([test, val, *train])}
    inner = [r for r in train if r != val]
    # 1) pick MLP config + debounce on the inner validation recording
    best = None
    sel_log = []
    for ci, cfg in enumerate(CONFIGS if scheme == "loro" else CONFIGS[:1]):
        m = fit_model([R[r].X[train_rows(R[r])] for r in inner], [R[r].lab[train_rows(R[r])] for r in inner], cfg)
        for k in DEBOUNCE:
            f1, hs_f1, _ = quick_score(*ml_outputs(m, R[val], k), R[val])
            sel_log.append(dict(cfg=ci, k=k, val_macro_f1=f1, val_hs_f1=hs_f1))
            score = (np.nan_to_num(hs_f1) + f1) / 2
            if best is None or score > best[0]:
                best = (score, ci, k, m)
    _, ci, k, inner_model = best
    cfg = CONFIGS[ci]
    # 2) pick a fine-tuning protocol per (source, x) on the same validation recording
    ft_choice = {}
    for src in SOURCES:
        for x in X_LIST:
            scores = []
            for pi, proto in enumerate(FT_PROTOCOLS):
                mf = finetune(inner_model, R[val], x, src, proto)
                f1, hs_f1, _ = quick_score(*ml_outputs(mf, R[val], k), R[val])
                scores.append((np.nan_to_num(hs_f1) + f1) / 2)
            ft_choice[f"{src}_{x}"] = int(np.argmax(scores))
    # 3) final model on all training recordings, scored on the test recording
    model = fit_model([R[r].X[train_rows(R[r])] for r in train], [R[r].lab[train_rows(R[r])] for r in train], cfg)
    T = R[test]
    out = {"scheme": scheme, "test": test, "train": train, "val": val, "cfg": ci, "debounce": k,
           "selection": sel_log, "ft_choice": ft_choice, "n_params": model.n_params(), "macs": model.macs(),
           "runs": {}}
    def add(name, ph, ev):
        out["runs"][name] = pack(ph, ev, T)
        g = M.cycle_gate(ph, T.tg)
        out["runs"][name + "+gate"] = pack(g, M.events_from_phases(g, T.tg), T)

    add("ml", *ml_outputs(model, T, k))
    for src in SOURCES:
        for x in X_LIST:
            mf = finetune(model, T, x, src, FT_PROTOCOLS[ft_choice[f"{src}_{x}"]])
            add(f"online_{src}_{x}", *ml_outputs(mf, T, k))

    # other model families, and the MLP with one sensor only; debounce picked on the same inner validation
    def pick_k(predict_val):
        raw = predict_val(R[val])
        return max(DEBOUNCE, key=lambda kk: (lambda f, h, _: (np.nan_to_num(h) + f) / 2)(
            *quick_score(M.debounce(raw, kk), M.events_from_phases(M.debounce(raw, kk), R[val].tg), R[val])))

    def rows(recs, step=1):
        X = np.concatenate([R[r].X[train_rows(R[r])][::step] for r in recs])
        y = np.concatenate([R[r].lab[train_rows(R[r])][::step] for r in recs])
        return X, y

    def adapt(name, m, make, x, src):
        """Online learning for a non-neural model, in the way each model allows."""
        r, y = finetune_labels(T, x, src)
        if len(r) < 50 or set(np.unique(y)) != {0, 1, 2, 3}:
            return m
        Xft = T.X[r]
        if name == "hgb":    # 50 more boosting rounds fitted to the new walker
            m2 = copy.deepcopy(m)
            m2.set_params(warm_start=True, early_stopping=False, max_iter=m.n_iter_ + 50)
            return m2.fit(Xft, y)
        if name == "rf":     # 50 more trees grown on the new walker
            m2 = copy.deepcopy(m)
            m2.set_params(warm_start=True, n_estimators=m.n_estimators + 50)
            return m2.fit(Xft, y)
        Xt, yt = rows(train, 3)  # linear: refit with the new walker weighted as much as all training data
        w = np.r_[np.ones(len(yt)), np.full(len(y), len(yt) / len(y))]
        step = list(m.named_steps)[-1]
        return make().fit(np.r_[Xt, Xft], np.r_[yt, y], **{f"{step}__sample_weight": w})

    for name, make in ZOO.items():
        st = ZOO_STRIDE.get(name, 1)
        m_in = make().fit(*rows(inner, st))
        kk = pick_k(lambda RR: m_in.predict(RR.X))
        m = make().fit(*rows(train, st))
        ph = M.debounce(m.predict(T.X).astype(int), kk)
        add(name, ph, M.events_from_phases(ph, T.tg))
        out.setdefault("zoo_debounce", {})[name] = kk
        for src in SOURCES:
            for x in X_LIST:
                ma = adapt(name, m, make, x, src)
                ph = M.debounce(ma.predict(T.X).astype(int), kk)
                add(f"{name}_online_{src}_{x}", ph, M.events_from_phases(ph, T.tg))
    for sens, keys in SENSORS.items():
        cols = [i for i, n in enumerate(T.names) if n.startswith(keys)]
        Xi, yi = rows(inner)
        m_in = M.MLP(len(cols), cfg["hidden"]).fit(Xi[:, cols], yi, epochs=EPOCHS,
                                                   class_weight=M.balanced_weights(yi) if cfg["balanced"] else None)
        kk = pick_k(lambda RR: m_in.predict(RR.X[:, cols]))
        Xt, yt = rows(train)
        m = M.MLP(len(cols), cfg["hidden"]).fit(Xt[:, cols], yt, epochs=EPOCHS,
                                                class_weight=M.balanced_weights(yt) if cfg["balanced"] else None)
        ph = M.debounce(m.predict(T.X[:, cols]), kk)
        add(f"mlp_{sens}", ph, M.events_from_phases(ph, T.tg))
        for src in SOURCES:
            for x in X_LIST:
                mf = finetune(m, T, x, src, FT_PROTOCOLS[ft_choice[f"{src}_{x}"]], cols=cols)
                ph = M.debounce(mf.predict(T.X[:, cols]), kk)
                add(f"mlp_{sens}_online_{src}_{x}", ph, M.events_from_phases(ph, T.tg))
    return out


def pack(ph, ev, R):
    return dict(cm=S.confusion(ph, R.lab, R.test_mask).tolist(), events=S.event_scores(ev, R.ev),
                phase=ph.astype(np.int8), ev=ev)


def run_fsm_all(R):
    thr = M.walk_cal(R, sd.CAL_S)
    ph, ev = M.run_fsm(R, thr, R.walk0 + sd.CAL_S)
    out = pack(ph, ev, R)
    out["thr"] = thr
    return out


def jobs():
    js = []
    for i, r in enumerate(RECS):
        train = [x for x in RECS if x != r]
        val = train[i % len(train)]
        js.append(dict(scheme="loro", test=r, train=train, val=val))
        other = [x for x in RECS if sd.PERSON[x] != sd.PERSON[r]]
        js.append(dict(scheme="lopo", test=r, train=other, val=other[i % len(other)]))
    return js


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    results = {"fsm": {}, "folds": []}
    for r in RECS:
        results["fsm"][r] = run_fsm_all(prep(r))
    print(f"FSM done {time.time() - t0:.0f}s", flush=True)
    with ProcessPoolExecutor(6) as ex:
        for res in ex.map(run_fold, jobs()):
            results["folds"].append(res)
            print(f"{res['scheme']} {res['test']} cfg {res['cfg']} k {res['debounce']} "
                  f"{time.time() - t0:.0f}s", flush=True)
    np.save(OUT / "raw_results.npy", results, allow_pickle=True)
    print("saved", OUT / "raw_results.npy")
