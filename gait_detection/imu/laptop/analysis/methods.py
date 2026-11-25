"""The detectors under test, all fed the same 100 Hz grid signals.

  FSM        the board's GaitFsm.h (compiled from SAKER_Sender), thresholds from
             the board's analyseWalk() on the first CAL seconds of walk_normal.
  ML         a small MLP over causal features (only the current and past samples).
  ML+online  the same MLP fine-tuned on the first x seconds of walk_normal of the
             recording it is then tested on.
"""
import subprocess
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import saker_data as sd

HERE = Path(__file__).resolve().parent
FSM_BIN, CAL_BIN = HERE / "fsm_replay", HERE / "walk_cal"
EVENT_TYPE = {1: "PO", 2: "TO", 3: "MS", 4: "HS"}   # GaitEvent enum -> mark type; 5 = TIMEOUT
PHASE_EVENT = {1: "PO", 2: "TO", 3: "MS", 0: "HS"}  # entering this phase = this event


class Rec:
    """One recording on the regular 100 Hz grid (gaps filled by straight lines)."""

    def __init__(self, rec):
        d = sd.load(rec)
        self.rec, self.d = rec, d
        self.tg, self.received, self.sig, self.task = sd.grid(
            d, keys=("shank_sag", "thigh_sag", "thigh_mag", "shank_acc_dyn"))
        gi = d["gi"]
        self.shank3 = np.stack([np.interp(self.tg, gi / sd.FS, d["shank_c"][:, k]) for k in range(3)], 1)
        self.thigh3 = np.stack([np.interp(self.tg, gi / sd.FS, d["thigh_c"][:, k]) for k in range(3)], 1)
        self.walk0, self.setup_end = d["walk0"], d["setup_end"]
        self.n = len(self.tg)

    def idx(self, t):
        return int(np.ceil(t * sd.FS - 1e-6))


# ---------------------------------------------------------------- FSM

def walk_cal(R, seconds):
    """analyseWalk() on the first `seconds` of walk_normal. Returns dict of thresholds."""
    a, b = R.idx(R.walk0), R.idx(R.walk0 + seconds)
    lines = "\n".join(" ".join(f"{v:.3f}" for v in np.r_[R.shank3[i], R.thigh3[i]]) for i in range(a, b))
    out = subprocess.run([str(CAL_BIN)], input=lines, capture_output=True, text=True, check=True).stdout.split()
    res = dict(status=out[0], swingPeak=float(out[1]), pushOffPeak=float(out[2]), swingEnter=float(out[3]),
               pushOffEnter=float(out[4]), heelStrike=float(out[5]), steps=int(out[6]), shank_axis=out[7])
    if res["shank_axis"] != R.d["shank_axis"]:
        raise RuntimeError(f"{R.rec}: {seconds}s walk cal picked {res['shank_axis']}, not {R.d['shank_axis']}")
    return res


def run_fsm(R, thr, t0, t1=None):
    """Board FSM from t0 (reset there) to t1. Returns (phase per grid sample, -1 outside; [(t, type)])."""
    a = R.idx(t0)
    b = R.n if t1 is None else R.idx(t1)
    s, m = R.sig["shank_sag"], R.sig["thigh_mag"]
    lines = "\n".join(f"{int(round(R.tg[i] * 1000))} {s[i]:.3f} {m[i]:.3f}" for i in range(a, b))
    args = [str(FSM_BIN), f"{thr['swingEnter']}", f"{thr['pushOffEnter']}", f"{thr['heelStrike']}"]
    out = np.array(subprocess.run(args, input=lines, capture_output=True, text=True, check=True).stdout.split(),
                   int).reshape(-1, 2)
    phase = np.full(R.n, -1, int)
    phase[a:b] = out[:, 0]
    events = [(R.tg[a + i], EVENT_TYPE[e]) for i, e in enumerate(out[:, 1]) if e in EVENT_TYPE]
    return phase, events


# ---------------------------------------------------------------- features

LAGS = (1, 2, 3, 5, 8, 12, 16, 20)
W_SHORT, W_LONG = 20, (50, 100)


def _lag(x, k):
    return np.concatenate([np.full(k, x[0]), x[:-k]])


def _win(x, w):
    return sliding_window_view(np.concatenate([np.full(w - 1, x[0]), x]), w)


def features(R):
    """Causal features per grid sample: only the current and the previous 99 samples."""
    cols, names = [], []
    for key in ("shank_sag", "thigh_sag", "shank_acc_dyn", "thigh_mag"):
        x = R.sig[key]
        win = _win(x, W_SHORT)
        block = [x] + [x - _lag(x, k) for k in LAGS] + [
            win.min(1), win.max(1), win.mean(1), win.std(1),
            (W_SHORT - 1 - win.argmax(1)).astype(float), (W_SHORT - 1 - win.argmin(1)).astype(float)]
        cols += block
        names += [key] + [f"{key}_d{k}" for k in LAGS] + [f"{key}_{s}{W_SHORT}" for s in
                                                          ("min", "max", "mean", "std", "since_max", "since_min")]
        if key in ("shank_sag", "thigh_sag"):
            for w in W_LONG:
                lw = _win(x, w)
                cols += [lw.min(1), lw.max(1), (w - 1 - lw.argmax(1)).astype(float),
                         (w - 1 - lw.argmin(1)).astype(float)]
                names += [f"{key}_{s}{w}" for s in ("min", "max", "since_max", "since_min")]
    return np.stack(cols, 1).astype(np.float32), names


# ---------------------------------------------------------------- MLP (numpy, so fine-tuning is fully controlled)

class MLP:
    def __init__(self, n_in, hidden=(32, 16), n_out=4, seed=0):
        rng = np.random.default_rng(seed)
        sizes = [n_in, *hidden, n_out]
        self.W = [rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.mu = np.zeros(n_in, np.float32)
        self.sd = np.ones(n_in, np.float32)

    def copy(self):
        m = MLP.__new__(MLP)
        m.W = [w.copy() for w in self.W]
        m.b = [b.copy() for b in self.b]
        m.mu, m.sd = self.mu.copy(), self.sd.copy()
        return m

    def n_params(self):
        return sum(w.size + b.size for w, b in zip(self.W, self.b))

    def macs(self):
        return sum(w.size for w in self.W)

    def _forward(self, X):
        acts = [(X - self.mu) / self.sd]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ w + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def proba(self, X, batch=65536):
        out = []
        for i in range(0, len(X), batch):
            z = self._forward(X[i:i + batch])[-1]
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            out.append(e / e.sum(1, keepdims=True))
        return np.concatenate(out)

    def predict(self, X):
        return self.proba(X).argmax(1)

    def fit(self, X, y, epochs=30, lr=1e-3, batch=256, l2=1e-4, class_weight=None, seed=0,
            anchor=None, anchor_l2=0.0, set_scaler=True):
        """Adam on softmax cross-entropy. anchor: weights to stay close to (L2-SP) when fine-tuning."""
        rng = np.random.default_rng(seed)
        if set_scaler:
            self.mu = X.mean(0)
            self.sd = X.std(0) + 1e-6
        cw = np.ones(4, np.float32) if class_weight is None else np.asarray(class_weight, np.float32)
        params = self.W + self.b
        m1 = [np.zeros_like(p) for p in params]
        m2 = [np.zeros_like(p) for p in params]
        step = 0
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for i in range(0, len(X), batch):
                j = order[i:i + batch]
                acts = self._forward(X[j])
                z = acts[-1] - acts[-1].max(1, keepdims=True)
                p = np.exp(z)
                p /= p.sum(1, keepdims=True)
                wts = cw[y[j]]
                g = p.copy()
                g[np.arange(len(j)), y[j]] -= 1
                g *= (wts / wts.sum())[:, None]
                gW, gb = [None] * len(self.W), [None] * len(self.W)
                for k in range(len(self.W) - 1, -1, -1):
                    gW[k] = acts[k].T @ g + l2 * self.W[k]
                    if anchor is not None:
                        gW[k] += anchor_l2 * (self.W[k] - anchor.W[k])
                    gb[k] = g.sum(0)
                    if k:
                        g = (g @ self.W[k].T) * (acts[k] > 0)
                step += 1
                for p_, g_, a1, a2 in zip(params, gW + gb, m1, m2):
                    a1 *= 0.9
                    a1 += 0.1 * g_
                    a2 *= 0.999
                    a2 += 0.001 * g_ * g_
                    p_ -= lr * (a1 / (1 - 0.9 ** step)) / (np.sqrt(a2 / (1 - 0.999 ** step)) + 1e-8)
        return self


def balanced_weights(y):
    counts = np.bincount(y, minlength=4).astype(float)
    return (counts.sum() / (4 * np.maximum(counts, 1))) ** 0.5


# ---------------------------------------------------------------- phase sequence -> events

def debounce(pred, k):
    """A new phase is accepted after k consecutive samples agree (k=1: no debounce)."""
    if k <= 1:
        return pred.copy()
    out = np.empty_like(pred)
    cur, cand, cnt = pred[0], -1, 0
    for i, p in enumerate(pred):
        if p == cur:
            cnt = 0
        else:
            if p == cand:
                cnt += 1
            else:
                cand, cnt = p, 1
            if cnt >= k:
                cur, cnt = p, 0
        out[i] = cur
    return out


def events_from_phases(phase, tg, t0=None):
    """Events at confirmed phase changes, in the board's order:
    ->PRE_SWING from STANCE = PO; ->SWING from STANCE/PRE_SWING = TO;
    ->TERMINAL_SWING from SWING = MS; ->STANCE from SWING/TERMINAL_SWING = HS."""
    ok = {1: (0,), 2: (0, 1), 3: (2,), 0: (2, 3)}
    ch = np.flatnonzero(phase[1:] != phase[:-1]) + 1
    ev = []
    for i in ch:
        a, b = phase[i - 1], phase[i]
        if a >= 0 and b >= 0 and a in ok[b] and (t0 is None or tg[i] >= t0):
            ev.append((tg[i], PHASE_EVENT[b]))
    return ev


# ---------------------------------------------------------------- one heel strike per stride

CYCLE_NEXT = {0: (1, 2), 1: (2,), 2: (3, 0), 3: (0,)}   # forward steps only (PO may be skipped)


def cycle_gate(pred, tg, refractory=0.3, reset_samples=20):
    """Lets the ML phase move only forward round the gait cycle, like the board's FSM.
    A heel strike (-> STANCE) needs a swing since the last one and >= `refractory` s
    since it. If the model has said STANCE for `reset_samples` in a row, the gate
    falls back to STANCE whatever state it is in (as the FSM's timeouts do)."""
    out = np.empty_like(pred)
    cur, last_hs, swung, run0 = pred[0], -1e9, False, 0
    for i, p in enumerate(pred):
        run0 = run0 + 1 if p == 0 else 0
        if p != cur:
            ok = p in CYCLE_NEXT.get(cur, ())
            if ok and p == 0 and (not swung or tg[i] - last_hs < refractory):
                ok = False
            if not ok and p == 0 and run0 >= reset_samples:
                ok = True
            if ok:
                if p == 0:
                    last_hs, swung = tg[i], False
                elif p == 2:
                    swung = True
                cur = p
        out[i] = cur
    return out
