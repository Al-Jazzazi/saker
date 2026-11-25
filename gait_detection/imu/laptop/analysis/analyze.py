"""Tables, confidence intervals and agreement numbers from benchmark.py's output.

Usage: analyze.py  -> results/benchmark/summary.json (+ CSV tables, figures/*.png)
Confidence intervals: block bootstrap, 10 s blocks resampled within each
recording (2000 draws), paired across methods (same blocks for every method).
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import labels as lb  # noqa: E402
import methods as M  # noqa: E402
import saker_data as sd  # noqa: E402
import scoring as S  # noqa: E402
from benchmark import OUT, RECS, SOURCES, X_LIST  # noqa: E402

TASKS = ["walk_normal", "walk_slow", "walk_fast", "turns", "stop_go", "stand"]
BLOCK_S, N_BOOT = 10.0, 2000
WIDE_S = 0.30   # second, looser match window: tells "late" apart from "missed"


class Lite:
    """What scoring needs from a recording (no features)."""

    def __init__(self, rec):
        R = M.Rec(rec)
        self.rec, self.tg, self.task, self.received = rec, R.tg, R.task, R.received
        self.marks = lb.load_marks(rec)
        self.lab = lb.phase_labels(self.marks, self.tg)
        self.setup_end = R.setup_end
        self.test_mask = (self.tg >= R.setup_end) & R.received
        self.ev = S.event_sets(self.marks, R.setup_end, self.tg[-1] + 0.01)
        self.block = np.floor((self.tg - R.setup_end) / BLOCK_S).astype(int)
        self.irregular = lb.load_marks(rec).get("irregular", [])

    def task_at(self, t):
        return self.task[np.clip(np.rint(np.asarray(t) * sd.FS).astype(int), 0, len(self.task) - 1)]


def method_runs(raw):
    """{method name: {rec: run}} for every method variant."""
    runs = {"fsm": dict(raw["fsm"])}
    for f in raw["folds"]:
        for k, v in f["runs"].items():
            runs.setdefault(f"{k}@{f['scheme']}", {})[f["test"]] = v
    return runs


def restrict(S_ev, keep_t):
    """Event set limited to manual events whose time passes keep_t(t)."""
    out = dict(S_ev)
    for key in ("sure", "unsure", "edge"):
        out[key] = {k: v[keep_t(v)] if len(v) else v for k, v in S_ev[key].items()}
    return out


def scores_for(run, L, task=None):
    mask = L.test_mask.copy()
    ev_pred = run["ev"]
    S_ev = L.ev
    if task is not None:
        mask &= L.task == task
        S_ev = restrict(L.ev, lambda v: L.task_at(v) == task)
        ev_pred = [(t, ty) for t, ty in ev_pred if L.task_at(t) == task]
    cm = S.confusion(run["phase"].astype(int), L.lab, mask)
    return cm, S.event_scores(ev_pred, S_ev)


def pooled(runs_m, Ls, task=None):
    cms, evs, wide = [], [], []
    for r, run in runs_m.items():
        cm, ev = scores_for(run, Ls[r], task)
        cms.append(cm)
        evs.append(ev)
        if task is None:
            wide.append(S.event_scores(run["ev"], Ls[r].ev, window=WIDE_S))
    cm = np.sum(cms, 0)
    out = dict(phase=S.phase_scores(cm), phase3=S.merge3(cm), events=S.summarize_events(evs), cm=cm.tolist())
    if wide:
        out["events_wide"] = S.summarize_events(wide)
    return out


def block_stats(run, L):
    """Per 10 s test block: confusion matrix and HS/TO/MS hit/true/false-alarm counts."""
    nb = L.block[L.test_mask].max() + 1
    cms = np.zeros((nb, 4, 4), int)
    m = L.test_mask & (L.lab >= 0)
    np.add.at(cms, (L.block[m], L.lab[m], run["phase"].astype(int)[m]), 1)
    ev = np.zeros((nb, 4, 3), int)   # type x (hits, n_true, false alarms)
    sc = S.event_scores(run["ev"], L.ev)
    for ti, k in enumerate(S.TYPES):
        tt = L.ev["sure"][k]
        bt = np.floor((tt - L.setup_end) / BLOCK_S).astype(int)
        np.add.at(ev[:, ti, 1], bt[bt < nb], 1)
        # hits: re-derive which true events were matched
        pairs = S._match(tt, np.array(sorted(t for t, ty in run["ev"]
                                             if ty == k and t >= L.setup_end - S.MATCH_S)))
        hb = bt[[i for i, _ in pairs]] if pairs else np.array([], int)
        np.add.at(ev[:, ti, 0], hb[hb < nb], 1)
        fb = np.floor((np.asarray(sc[k]["fa_times"]) - L.setup_end) / BLOCK_S).astype(int)
        fb = fb[(fb >= 0) & (fb < nb)]
        np.add.at(ev[:, ti, 2], fb, 1)
    return cms, ev


def macro_f1(cm):
    return S.phase_scores(cm)["macro_f1"]


def bootstrap(all_runs, Ls, methods, seed=0):
    recs = [r for r in Ls if all(r in all_runs[m] for m in methods)]
    rng = np.random.default_rng(seed)
    stats = {m: {r: block_stats(all_runs[m][r], Ls[r]) for r in recs if r in all_runs[m]} for m in methods}
    draws = {m: {"macro_f1": [], "HS_recall": [], "HS_fa100": [], "TO_recall": [], "MS_recall": []} for m in methods}
    nblocks = {r: stats[methods[0]][r][0].shape[0] for r in recs}
    for _ in range(N_BOOT):
        pick = {r: rng.integers(0, nblocks[r], nblocks[r]) for r in recs}
        for m in methods:
            cm = sum(stats[m][r][0][pick[r]].sum(0) for r in recs)
            ev = sum(stats[m][r][1][pick[r]].sum(0) for r in recs)
            draws[m]["macro_f1"].append(macro_f1(cm))
            draws[m]["HS_recall"].append(ev[0, 0] / max(ev[0, 1], 1))
            draws[m]["HS_fa100"].append(100 * ev[0, 2] / max(ev[0, 1], 1))
            draws[m]["TO_recall"].append(ev[2, 0] / max(ev[2, 1], 1))
            draws[m]["MS_recall"].append(ev[3, 0] / max(ev[3, 1], 1))
    ci = {m: {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in d.items()}
          for m, d in draws.items()}
    diffs = {}
    for a in methods:
        for b in methods:
            if a < b:
                diffs[f"{a} - {b}"] = {k: [float(np.percentile(np.subtract(draws[a][k], draws[b][k]), q))
                                           for q in (2.5, 50, 97.5)] for k in draws[a]}
    return ci, diffs


# ---------------------------------------------------------------- annotation agreement

def agreement():
    """Blind double annotation (A vs B) and how much the reviewer changed (A vs reviewed)."""
    out = {}
    pairs = {"A_vs_B": ("raw-A", "raw-B"), "A_vs_reviewed": ("raw-A", "reviewed"), "B_vs_reviewed": ("raw-B", "reviewed")}
    for name, (ka, kb) in pairs.items():
        fa = {c["id"]: p for c, p in lb.chunk_files(ka)}
        fb = {c["id"]: p for c, p in lb.chunk_files(kb)}
        common = sorted(set(fa) & set(fb))
        res = {k: dict(n_a=0, n_b=0, matched=0, dt=[]) for k in S.TYPES}
        agree = tot = 0
        for cid in common:
            ma, mb = json.load(open(fa[cid])), json.load(open(fb[cid]))
            c = next(c for c, _ in lb.chunk_files(ka) if c["id"] == cid)
            for k in S.TYPES:
                ta = np.array(sorted(e["t"] for e in ma["events"] if e["type"] == k and c["start"] <= e["t"] < c["end"]))
                tb = np.array(sorted(e["t"] for e in mb["events"] if e["type"] == k and c["start"] <= e["t"] < c["end"]))
                pr = S._match(ta, tb)
                res[k]["n_a"] += len(ta)
                res[k]["n_b"] += len(tb)
                res[k]["matched"] += len(pr)
                res[k]["dt"] += [1000 * (tb[j] - ta[i]) for i, j in pr]
            tg = np.arange(int(c["start"] * sd.FS), int(c["end"] * sd.FS)) / sd.FS
            la, lbb = lb.phase_labels(ma, tg), lb.phase_labels(mb, tg)
            both = (la >= 0) & (lbb >= 0)
            agree += int((la[both] == lbb[both]).sum())
            tot += int(both.sum())
        for k in S.TYPES:
            d = np.abs(res[k]["dt"])
            res[k] = dict(n_a=res[k]["n_a"], n_b=res[k]["n_b"], matched=res[k]["matched"],
                          abs_dt_median_ms=float(np.median(d)) if len(d) else None,
                          abs_dt_p90_ms=float(np.percentile(d, 90)) if len(d) else None,
                          signed_dt_mean_ms=float(np.mean(res[k]["dt"])) if len(d) else None)
        out[name] = dict(chunks=common, events=res, phase_agreement=agree / tot if tot else None, n_samples=tot)
    return out


def rule_vs_manual(recs=RECS):
    """QA only: how far the old rule labeller (gait_labels.py style, whole recording,
    looks ahead) lands from the manual marks. Never used as ground truth."""
    from benchmark import lookahead_labels
    res = {k: dict(n_manual=0, n_rule=0, matched=0, dt=[]) for k in S.TYPES}
    for r in recs:
        R = M.Rec(r)
        lab = lookahead_labels(R, 0, R.n)
        ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        rule = {k: [] for k in S.TYPES}
        for i in ch:
            if lab[i] >= 0:
                rule[["HS", "PO", "TO", "MS"][lab[i]]].append(R.tg[i])
        marks = lb.load_marks(r)
        for k in S.TYPES:
            tm = np.array(sorted(e["t"] for e in marks["events"] if e["type"] == k and e.get("conf") != "unsure"))
            tr = np.array(sorted(rule[k]))
            pr = S._match(tm, tr)
            res[k]["n_manual"] += len(tm)
            res[k]["n_rule"] += len(tr)
            res[k]["matched"] += len(pr)
            res[k]["dt"] += [1000 * (tr[j] - tm[i]) for i, j in pr]
    for k in S.TYPES:
        d = np.asarray(res[k].pop("dt"))
        res[k].update(median_ms=float(np.median(d)) if len(d) else None,
                      abs_p90_ms=float(np.percentile(np.abs(d), 90)) if len(d) else None)
    return res


def write_table(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def flat(name, rec, p):
    row = dict(method=name, recording=rec, macro_f1=round(p["phase"]["macro_f1"], 4), acc=round(p["phase"]["acc"], 4),
               macro_f1_3phase=round(p["phase3"]["macro_f1"], 4), n_samples=p["phase"]["n"])
    for i, ph in enumerate(lb.PHASES):
        row[f"f1_{ph}"] = round(p["phase"]["f1"][i], 4)
    for k in S.TYPES:
        e = p["events"][k]
        if "events_wide" in p:
            row[f"{k}_recall_300ms"] = round(p["events_wide"][k]["recall"], 4)
            row[f"{k}_err_median_ms_300"] = round(p["events_wide"][k]["err_median_ms"], 1)
        row.update({f"{k}_n": e["n_true"], f"{k}_recall": round(e["recall"], 4), f"{k}_fa_per_100": round(e["fa_per_100"], 2),
                    f"{k}_f1": round(e["f1"], 4), f"{k}_err_median_ms": round(e["err_median_ms"], 1),
                    f"{k}_abs_err_p90_ms": round(e["abs_err_p90_ms"], 1), f"{k}_within50": round(e["within50"], 4),
                    f"{k}_fa_still": e["fa_still"]})
    return row


def main(raw, out=OUT, recs=RECS):
    Ls = {r: Lite(r) for r in recs}
    runs = method_runs(raw)
    for m, rm in runs.items():
        missing = set(recs) - set(rm)
        if missing:
            print(f"WARNING: {m} has no run for {sorted(missing)}")
    summary = {"methods": {}, "per_task": {}, "per_rec": {}, "per_person": {}}
    rows, task_rows = [], []
    for name, rm in runs.items():
        p = pooled(rm, Ls)
        summary["methods"][name] = p
        rows.append(flat(name, "ALL", p))
        for r in rm:
            pr = pooled({r: rm[r]}, Ls)
            summary["per_rec"].setdefault(name, {})[r] = pr
            rows.append(flat(name, r, pr))
        for person in ("yousef", "amjad"):
            sub = {r: v for r, v in rm.items() if sd.PERSON[r] == person}
            if not sub:
                continue
            pp = pooled(sub, Ls)
            summary["per_person"].setdefault(name, {})[person] = pp
            rows.append(flat(name, person, pp))
        for task in TASKS:
            pt = pooled(rm, Ls, task)
            summary["per_task"].setdefault(name, {})[task] = pt
            task_rows.append(flat(name, task, pt))
    write_table(out / "scores_by_method_and_recording.csv", rows)
    write_table(out / "scores_by_task.csv", task_rows)
    main = ["fsm", "ml@loro", "ml@lopo", "ml+gate@loro", "ml+gate@lopo"] + [f"online_{s}_{x}{g}@{sc}" for sc in ("loro", "lopo") for s in SOURCES for x in X_LIST for g in ("", "+gate")]
    main = [m for m in main if m in runs]
    summary["ci"], summary["paired_diff"] = bootstrap(runs, Ls, main)
    summary["agreement"] = agreement()
    summary["rule_vs_manual"] = rule_vs_manual(recs)
    summary["fsm_thresholds"] = {r: raw["fsm"][r]["thr"] for r in recs}
    summary["folds"] = [{k: f[k] for k in ("scheme", "test", "val", "cfg", "debounce", "ft_choice", "n_params", "macs")}
                        for f in raw["folds"]]
    summary["test_set"] = {r: dict(setup_end=float(Ls[r].setup_end), test_samples=int(Ls[r].test_mask.sum()),
                                   labelled=int((Ls[r].test_mask & (Ls[r].lab >= 0)).sum()),
                                   **{f"sure_{k}": int(len(Ls[r].ev["sure"][k])) for k in S.TYPES})
                           for r in recs}

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(out / "summary.json", "w") as f:
        json.dump(clean(summary), f, indent=1)
    print("wrote", out / "summary.json")
    return summary


if __name__ == "__main__":
    main(np.load(OUT / "raw_results.npy", allow_pickle=True).item())
