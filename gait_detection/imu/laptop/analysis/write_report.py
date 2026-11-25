"""Writes results/benchmark/REPORT.md (+ figures/) from summary.json.
Numbers in the text are filled from the data; the method and caveat sections are fixed text.
Usage: write_report.py"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark import OUT, SOURCES, X_LIST  # noqa: E402
from export_report import label  # noqa: E402

FIG = OUT / "figures"
TASKS = ["walk_normal", "walk_slow", "walk_fast", "turns", "stop_go", "stand"]


def pct(v):
    return "–" if v is None else f"{100 * v:.1f}%"


def ms(v):
    return "–" if v is None else f"{v:+.0f} ms"


def num(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def best_online(s, scheme, key=lambda p: p["phase"]["macro_f1"]):
    cands = [m for m in s["methods"] if m.startswith("online_") and m.endswith("@" + scheme) and "+gate" not in m]
    return max(cands, key=lambda m: key(s["methods"][m])) if cands else None


def main_methods(s):
    out = ["fsm", "ml@loro", "ml+gate@loro", "ml@lopo", "ml+gate@lopo"]
    for sch in ("loro", "lopo"):
        b = best_online(s, sch)
        if b:
            out.append(b)
        for src in ("fsm", "lookahead"):   # the realistic (no human labels) variants
            c = [m for m in s["methods"] if m.startswith(f"online_{src}_") and m.endswith("@" + sch) and "+gate" not in m]
            if c:
                out.append(max(c, key=lambda m: s["methods"][m]["phase"]["macro_f1"]))
    return [m for m in dict.fromkeys(out) if m in s["methods"]]


def figures(s, mains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    col = {"FSM": "#7a5195", "ML": "#1f77b4", "ML + online": "#2a9d8f"}

    # 1. macro F1 with bootstrap CI
    fig, ax = plt.subplots(figsize=(9, 0.5 + 0.45 * len(mains)))
    for i, m in enumerate(mains):
        p, ci = s["methods"][m], s["ci"].get(m, {}).get("macro_f1")
        v = p["phase"]["macro_f1"]
        ax.barh(i, v, color=col[label(m)[1]], height=0.6)
        if ci:
            ax.plot(ci, [i, i], color="k", lw=1.2)
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set_yticks(range(len(mains)), [label(m)[0] for m in mains], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("phase macro F1 on the shared test set (black = 95% block-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(FIG / "macro_f1.png", dpi=130)
    plt.close(fig)

    # 2. online learning curves
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, sch in zip(axes, ("loro", "lopo")):
        base = s["methods"].get(f"ml@{sch}")
        for src, c in zip(SOURCES, ("#2a9d8f", "#e76f51", "#264653")):
            xs, ys = [0], [base["phase"]["macro_f1"] if base else np.nan]
            for x in X_LIST:
                k = f"online_{src}_{x}@{sch}"
                if k in s["methods"]:
                    xs.append(x)
                    ys.append(s["methods"][k]["phase"]["macro_f1"])
            ax.plot(xs, ys, "o-", color=c, label=src, lw=2, ms=5)
        ax.axhline(s["methods"]["fsm"]["phase"]["macro_f1"], color="#7a5195", ls="--", lw=1.2, label="FSM")
        ax.set_title("trained on other recordings" if sch == "loro" else "trained on the other person", fontsize=10)
        ax.set_xlabel("seconds of the test person's normal walking used to fine-tune")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("phase macro F1")
    axes[1].legend(fontsize=8, title="fine-tune labels", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "online_curve.png", dpi=130)
    plt.close(fig)

    # 3. per task
    show = [m for m in mains[:4]]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    w = 0.8 / len(show)
    for j, m in enumerate(show):
        vals = [s["per_task"][m].get(t, {}).get("phase", {}).get("macro_f1") for t in TASKS]
        ax.bar(np.arange(len(TASKS)) + j * w, [v or 0 for v in vals], w * 0.92, label=label(m)[0],
               color=plt.cm.tab10(j))
    ax.set_xticks(np.arange(len(TASKS)) + 0.4 - w / 2, TASKS)
    ax.set_ylabel("phase macro F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("standing has only STANCE, so its macro F1 is capped at 0.25", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "per_task.png", dpi=130)
    plt.close(fig)


def zoo_table(s):
    names = ["ml", "logreg", "linsvm", "rf", "hgb", "mlp_shank", "mlp_thigh"]
    rows = ["| model (same features) | trained on | macro F1 | HS found | HS false / 100 | HS timing | TO found | macro F1 with gate | HS false / 100 with gate |",
            "|---|---|---|---|---|---|---|---|---|"]
    for sch in ("loro", "lopo"):
        for n in names:
            m, g = f"{n}@{sch}", f"{n}+gate@{sch}"
            if m not in s["methods"]:
                continue
            p, e = s["methods"][m], s["methods"][m]["events"]
            pg = s["methods"].get(g)
            rows.append(f"| {label(m)[0].rsplit(' (', 1)[0].replace('ML', 'MLP (both IMUs)') if n == 'ml' else label(m)[0].rsplit(' (', 1)[0]} "
                        f"| {'other recordings' if sch == 'loro' else 'other person'} | {num(p['phase']['macro_f1'])} "
                        f"| {pct(e['HS']['recall'])} | {num(e['HS']['fa_per_100'], 1)} | {ms(e['HS']['err_median_ms'])} "
                        f"| {pct(e['TO']['recall'])} | {num(pg['phase']['macro_f1']) if pg else '–'} "
                        f"| {num(pg['events']['HS']['fa_per_100'], 1) if pg else '–'} |")
    return "\n".join(rows)


def zoo_online_table(s):
    names = [("ml", "MLP (both IMUs)"), ("mlp_shank", "MLP, shank only"), ("mlp_thigh", "MLP, thigh only"),
             ("rf", "Random forest"), ("hgb", "Gradient boosting"), ("linsvm", "Linear SVM"), ("logreg", "Logistic regression")]
    rows = ["| model | trained on | no fine-tune | hand labels 12 s | hand labels 30 s | best self-labelled (rule / FSM) | false HS/100: none → hand 30 s |",
            "|---|---|---|---|---|---|---|"]
    for sch in ("loro", "lopo"):
        for n, nice in names:
            base = s["methods"].get(f"{n}@{sch}")
            if not base:
                continue
            key = (lambda src, x: f"online_{src}_{x}@{sch}") if n == "ml" else (lambda src, x: f"{n}_online_{src}_{x}@{sch}")
            g = lambda src, x: s["methods"].get(key(src, x))
            m12, m30 = g("manual", 12), g("manual", 30)
            selfs = [(src, x, g(src, x)) for src in ("lookahead", "fsm") for x in X_LIST if g(src, x)]
            bs = max(selfs, key=lambda t: t[2]["phase"]["macro_f1"]) if selfs else None
            rows.append(f"| {nice} | {'other recordings' if sch == 'loro' else 'other person'} | {num(base['phase']['macro_f1'])} "
                        f"| {num(m12['phase']['macro_f1']) if m12 else '–'} | {num(m30['phase']['macro_f1']) if m30 else '–'} "
                        f"| {num(bs[2]['phase']['macro_f1']) + f' ({bs[0]}, {bs[1]} s)' if bs else '–'} "
                        f"| {num(base['events']['HS']['fa_per_100'], 1)} → {num(m30['events']['HS']['fa_per_100'], 1) if m30 else '–'} |")
    return "\n".join(rows)


def size_section():
    import csv
    f = OUT / "size_study.csv"
    if not f.exists():
        return ""
    rows = list(csv.DictReader(open(f)))
    inp = [r for r in rows if r["study"] == "input"]
    mod = [r for r in rows if r["study"] == "model"]
    out = ["## 3c. Input size: how much past signal the model sees (new recording)", "",
           "| model | history | features | phase macro F1 | HS false / 100 | HS false / 100 with gate | input buffer |",
           "|---|---|---|---|---|---|---|"]
    out += [f"| {r['model']} | {r['history_ms']} ms | {r['n_features']} | {r['macro_f1']} | {r['HS_false_per_100']} "
            f"| {r['HS_false_per_100_gated']} | {r['input_buffer_bytes']} B |" for r in inp]
    out += ["", "## 3d. Model size (1 s history, new recording)", "",
            "| model | size | memory | work per prediction | phase macro F1 | HS false / 100 | with gate |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {r['model']} | {r['size']} | {r['memory_kB']} kB | {r['ops_per_prediction']} | {r['macro_f1']} "
            f"| {r['HS_false_per_100']} | {r['HS_false_per_100_gated']} |" for r in mod]
    out += ["", "Work per prediction = multiply-adds for the MLP, node comparisons for trees. The FSM needs a few comparisons "
            "and under 100 bytes. ESP32: ~520 kB RAM, a few MB flash. Tree nodes counted at 12 bytes each."]
    return "\n".join(out) + "\n"


def interpret_section():
    import csv
    f1, f2, f3 = OUT / "interpret_importance.csv", OUT / "interpret_top_features.csv", OUT / "interpret_errors.csv"
    if not f1.exists():
        return ""
    imp = list(csv.DictReader(open(f1)))
    out = ["## 3e. Interpretability: what the models rely on", "",
           "Permutation importance, leave one recording out: shuffle one group of inputs in the test recording and see how "
           "much phase macro F1 drops (mean over the 7 recordings, min–max in brackets).", "",
           "| input group | features | random forest | gradient boosting | neural net |", "|---|---|---|---|---|"]
    for g in dict.fromkeys(r["group"] for r in imp):
        cells = []
        for m in ("Random forest", "Gradient boosting", "Neural net (MLP)"):
            r = next(x for x in imp if x["group"] == g and x["model"] == m)
            cells.append(f"{float(r['macro_f1_drop']):.3f} [{float(r['macro_f1_drop_min']):.3f}–{float(r['macro_f1_drop_max']):.3f}]")
        n = next(x for x in imp if x["group"] == g)["n_features"]
        out.append(f"| {g} | {n} | " + " | ".join(cells) + " |")
    if f2.exists():
        out += ["", "Top features the random forest splits on (impurity importance, mean of 7 folds):", "",
                "| rank | feature | kind | importance |", "|---|---|---|---|"]
        out += [f"| {r['rank']} | `{r['feature']}` | {r['kind']} | {r['importance']} |" for r in csv.DictReader(open(f2))]
    if f3.exists():
        out += ["", "Where the errors are: share of wrongly labelled samples that lie within 100 ms of a true phase change "
                "(timing at the boundaries) rather than mid-phase (a wrong phase altogether).", "",
                "| method | wrong samples | within 100 ms of a phase change |", "|---|---|---|"]
        out += [f"| {r['method']} | {r['wrong_samples']} | {100 * float(r['share_within_100ms_of_phase_change']):.1f}% |"
                for r in csv.DictReader(open(f3))]
    return "\n".join(out) + "\n"


def table(s, mains):
    rows = ["| method | macro F1 [95% CI] | 3-phase F1 | HS found | HS false / 100 strides | HS timing (median) | HS \\|err\\| P90 | HS found ±300 ms | TO found | MS found | false HS while standing |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in mains:
        p, ci = s["methods"][m], s["ci"].get(m, {})
        e, w = p["events"], p.get("events_wide", p["events"])
        c = ci.get("macro_f1")
        rows.append(f"| {label(m)[0]} | {num(p['phase']['macro_f1'])} [{num(c[0]) if c else '–'}–{num(c[1]) if c else '–'}] "
                    f"| {num(p['phase3']['macro_f1'])} | {pct(e['HS']['recall'])} | {num(e['HS']['fa_per_100'], 1)} "
                    f"| {ms(e['HS']['err_median_ms'])} | {num(e['HS']['abs_err_p90_ms'], 0)} ms | {pct(w['HS']['recall'])} "
                    f"| {pct(e['TO']['recall'])} | {pct(e['MS']['recall'])} | {e['HS']['fa_still']} |")
    return "\n".join(rows)


def task_table(s, mains):
    rows = ["| method | " + " | ".join(TASKS) + " |", "|---|" + "---|" * len(TASKS)]
    for m in mains[:6]:
        cells = []
        for t in TASKS:
            p = s["per_task"][m].get(t)
            cells.append("–" if not p or not p["phase"]["n"] else
                         f"{num(p['phase']['macro_f1'], 2)} / HS {pct(p['events']['HS']['recall'])}")
        rows.append(f"| {label(m)[0]} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def rec_table(s, mains):
    recs = list(s["test_set"])
    rows = ["| method | " + " | ".join(recs) + " |", "|---|" + "---|" * len(recs)]
    for m in mains[:4]:
        rows.append(f"| {label(m)[0]} | " + " | ".join(
            num(s["per_rec"][m][r]["phase"]["macro_f1"], 2) if r in s["per_rec"][m] else "–" for r in recs) + " |")
    return "\n".join(rows)


def online_table(s):
    rows = ["| trained on | fine-tune labels | " + " | ".join(f"{x} s" for x in (0, *X_LIST)) + " |",
            "|---|---|" + "---|" * (len(X_LIST) + 1)]
    for sch in ("loro", "lopo"):
        for src in SOURCES:
            cells = []
            for x in (0, *X_LIST):
                k = f"ml@{sch}" if x == 0 else f"online_{src}_{x}@{sch}"
                p = s["methods"].get(k)
                cells.append("–" if not p else f"{num(p['phase']['macro_f1'], 3)} (HS F1 {num(p['events']['HS']['f1'], 2)})")
            rows.append(f"| {'other recordings' if sch == 'loro' else 'other person'} | {src} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def agreement_text(s):
    a = s["agreement"]
    out = []
    for comp, nice in (("A_vs_reviewed", "changes the reviewer made to the annotator's marks (not an independent agreement: the reviewer edits the annotator's file)"), ("A_vs_B", "two blind annotators")):
        d = a.get(comp)
        if not d or not d["chunks"]:
            continue
        parts = []
        for k in ("HS", "PO", "TO", "MS"):
            e = d["events"][k]
            if e["n_a"]:
                parts.append(f"{k}: {e['matched']}/{max(e['n_a'], e['n_b'])} matched, median |Δt| "
                             f"{num(e['abs_dt_median_ms'], 0)} ms (P90 {num(e['abs_dt_p90_ms'], 0)} ms)")
        out.append(f"- **{nice}** ({len(d['chunks'])} chunks; per-sample phase agreement {pct(d['phase_agreement'])}): "
                   + "; ".join(parts))
    return "\n".join(out) or "- (no second annotations available)"


def main():
    s = json.load(open(OUT / "summary.json"))
    mains = main_methods(s)
    figures(s, mains)
    fsm, ml, mlp = (s["methods"][k] for k in ("fsm", "ml@loro", "ml@lopo"))
    bo = best_online(s, "loro")
    bo_p = s["methods"][bo] if bo else None
    rank = sorted(mains, key=lambda m: -s["methods"][m]["phase"]["macro_f1"])
    hs_rank = sorted(mains, key=lambda m: -(s["methods"][m]["events"]["HS"]["f1"] or 0))
    n_test = sum(v["labelled"] for v in s["test_set"].values())
    n_hs = sum(v["sure_HS"] for v in s["test_set"].values())
    md = f"""# SAKER gait detection: FSM vs ML vs ML + online learning

Ground truth: every stride in the 7 real recordings marked **by eye** (heel strike, push-off, toe-off, mid-swing) on plots of the shank/thigh gyro and shank acceleration. No detection code placed any mark.
Test set (identical for every method): in each recording everything after **walk_normal start + 30 s**: {n_test:,} labelled samples, {n_hs:,} sure heel strikes.

## Headline

- Best phase accuracy: **{label(rank[0])[0]}** (macro F1 {num(s['methods'][rank[0]]['phase']['macro_f1'])}).
- Best heel-strike detection (the motor trigger): **{label(hs_rank[0])[0]}** (HS F1 {num(s['methods'][hs_rank[0]]['events']['HS']['f1'])}).
- FSM: macro F1 {num(fsm['phase']['macro_f1'])}, finds {pct(fsm['events']['HS']['recall'])} of heel strikes with {num(fsm['events']['HS']['fa_per_100'], 1)} false ones per 100 strides, median timing {ms(fsm['events']['HS']['err_median_ms'])}.
- ML (new recording): macro F1 {num(ml['phase']['macro_f1'])}; ML (new person): {num(mlp['phase']['macro_f1'])}.
- Best ML + online (new recording): {label(bo)[0] if bo else '–'}, macro F1 {num(bo_p['phase']['macro_f1']) if bo_p else '–'}.

![macro F1](figures/macro_f1.png)

## 1. Main comparison

{table(s, mains)}

"found" = matched one-to-one within ±150 ms of a manual mark of the same type; timing = detection minus manual mark (+ = late). The ±300 ms column separates *late* from *missed*. 3-phase F1 merges STANCE and PRE_SWING, so it does not depend on where push-off is put (the softest landmark).

## 1b. Different ML models and sensors (same features, same test set)

{zoo_table(s)}

"gate" = the model's phases may only move forward round the gait cycle, and a heel strike needs a swing since the last one and ≥ 300 ms gap (like the FSM).

## 2. By task (macro F1 / heel strikes found)

{task_table(s, mains)}

![per task](figures/per_task.png)

## 3. Online learning: how much does fine-tuning on the first x seconds help?

{online_table(s)}

`manual` = fine-tuned on the by-eye labels of those seconds (best case, needs a human). `fsm` = on the board FSM's own phases (what the board could do alone). `lookahead` = on an offline shank-gyro labeller run over the buffered seconds (also possible on the board).

![online](figures/online_curve.png)

### 3b. Online learning for every model (phase macro F1)

{zoo_online_table(s)}

How each model is adapted: neural nets are fine-tuned by gradient steps; gradient boosting gets 50 more boosting rounds fitted to the new walker; random forest gets 50 more trees grown on the new walker; linear models are refit with the new walker's seconds weighted as much as all training data.

{size_section()}
{interpret_section()}
## 4. Per recording (macro F1)

{rec_table(s, mains)}

## 5. How good is the ground truth?

{agreement_text(s)}

## 6. Method

- **Same test set for all methods.** The setup window (opening stand + first 30 s of walk_normal) is never scored. The FSM calibrates on its first 12 s (the board's `analyseWalk`), and ML + online fine-tunes on its first x ≤ 30 s. Plain ML never sees the test recording.
- **FSM** = the board's `GaitFsm.h`, compiled from `SAKER_Sender`, fed the same 100 Hz stream (Wi-Fi gaps filled by straight lines; the board itself never has gaps).
- **ML** = small MLP (≈3k weights) on causal features (current + past 1 s of shank/thigh rate, shank |acc|, thigh |gyro|), debounced phase changes → events. Trained leave-one-recording-out and leave-one-person-out; hyper-parameters picked on an inner validation recording, never the test one.
- **ML + online** = the same MLP fine-tuned (all layers, Adam, protocol chosen on the inner validation recording).
- Bootstrap: 10 s blocks resampled within each recording, 2000 draws, paired across methods.

## 7. Caveats

- 2 people, 7 recordings, one session per person and leg; Amjad wore slippers.
- Truth is by-eye on IMU traces, with no video or foot switch. Push-off is the least certain mark, and the FSM's push-off threshold (60% of the dip) sits later than the by-eye push-off by design.
- ~7-10% of samples are lost over Wi-Fi; events hidden by a gap are marked unclear and not scored.
- Annotation: 36 chunks by Opus agents; the last 14 completed by Sonnet at low effort (mostly verifying earlier Opus drafts). 1 chunk fully reviewed, 3 reviews stopped part-way; blind doubles cancelled to save credits.
"""
    extra = OUT / "INTERPRETATION.md"
    if extra.exists():
        md += "\n" + extra.read_text()
    (OUT / "REPORT.md").write_text(md)
    print("wrote", OUT / "REPORT.md")


if __name__ == "__main__":
    main()
