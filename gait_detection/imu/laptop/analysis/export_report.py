"""Flattens results/benchmark/summary.json into the CSV files the report page reads.
Usage: export_report.py [DIR]  -> DIR/report/*.csv (DIR defaults to results/benchmark)"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saker_data as sd  # noqa: E402
from benchmark import OUT, SOURCES, X_LIST  # noqa: E402

REP = OUT / "report"
TYPES = ["HS", "PO", "TO", "MS"]
PHASES = ["STANCE", "PRE_SWING", "SWING", "TERMINAL_SWING"]
SRC_LABEL = {"manual": "manual labels", "fsm": "FSM pseudo-labels", "lookahead": "look-ahead pseudo-labels"}


def label(m):
    if "+gate" in m:
        lab, fam, scheme, src, x = label(m.replace("+gate", ""))
        return lab.replace("ML", "ML + gate", 1), fam, scheme, src, x
    if m == "fsm":
        return "FSM (12 s calibration)", "FSM", "-", "", ""
    name, scheme = m.split("@")
    sch = "new recording" if scheme == "loro" else "new person"
    if "_online_" in name:
        base, rest = name.split("_online_")
        src, x = rest.split("_")
        blab = label(f"{base}@{scheme}")[0].rsplit(" (", 1)[0]
        return f"{blab} + online, {SRC_LABEL[src]}, {x} s ({sch})", "ML + online", scheme, src, x
    zoo = {"logreg": "Logistic regression", "linsvm": "Linear SVM", "rf": "Random forest",
           "hgb": "Gradient boosting", "mlp_shank": "MLP, shank IMU only", "mlp_thigh": "MLP, thigh IMU only"}
    if name in zoo:
        return f"{zoo[name]} ({sch})", "ML", scheme, "", ""
    if name == "ml":
        return f"ML ({sch})", "ML", scheme, "", ""
    _, src, x = name.split("_")
    return f"ML + online, {SRC_LABEL[src]}, {x} s ({sch})", "ML + online", scheme, src, x


def r4(v):
    return None if v is None else round(v, 4)


def r1(v):
    return None if v is None else round(v, 1)


def write(name, rows):
    REP.mkdir(parents=True, exist_ok=True)
    with open(REP / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(name, len(rows))


def ev_cols(p, prefix=""):
    out = {}
    for k in TYPES:
        e = p["events"][k]
        out.update({f"{k}_recall": r4(e["recall"]), f"{k}_fa_per_100": r4(e["fa_per_100"]),
                    f"{k}_f1": r4(e["f1"]), f"{k}_err_median_ms": r1(e["err_median_ms"]),
                    f"{k}_abs_err_p90_ms": r1(e["abs_err_p90_ms"]), f"{k}_within50": r4(e["within50"]),
                    f"{k}_n": e["n_true"]})
        if "events_wide" in p:
            out[f"{k}_recall_300ms"] = r4(p["events_wide"][k]["recall"])
            out[f"{k}_err_median_ms_300"] = r1(p["events_wide"][k]["err_median_ms"])
    return out


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
        REP = OUT / "report"
    s = json.load(open(OUT / "summary.json"))
    methods = list(s["methods"])
    rows = []
    for m in methods:
        p = s["methods"][m]
        lab, fam, scheme, src, x = label(m)
        ci = s["ci"].get(m, {})
        row = dict(method=m, label=lab, family=fam, scheme=scheme, source=src, x_s=x,
                   macro_f1=r4(p["phase"]["macro_f1"]), macro_f1_lo=r4(ci.get("macro_f1", [None])[0]),
                   macro_f1_hi=r4(ci.get("macro_f1", [None, None])[1]), macro_f1_3phase=r4(p["phase3"]["macro_f1"]),
                   accuracy=r4(p["phase"]["acc"]),
                   HS_recall_lo=r4(ci.get("HS_recall", [None])[0]), HS_recall_hi=r4(ci.get("HS_recall", [None, None])[1]),
                   HS_fa100_lo=r4(ci.get("HS_fa100", [None])[0]), HS_fa100_hi=r4(ci.get("HS_fa100", [None, None])[1]),
                   HS_false_alarms_while_still=p["events"]["HS"]["fa_still"], n_samples=p["phase"]["n"])
        row.update({f"f1_{ph}": r4(p["phase"]["f1"][i]) for i, ph in enumerate(PHASES)})
        row.update(ev_cols(p))
        rows.append(row)
    write("methods.csv", rows)

    write("per_task.csv", [dict(method=m, label=label(m)[0], task=t, macro_f1=r4(p["phase"]["macro_f1"]),
                                accuracy=r4(p["phase"]["acc"]), n_samples=p["phase"]["n"], **ev_cols(p))
                           for m in methods for t, p in s["per_task"][m].items() if p["phase"]["n"] > 0])

    write("per_recording.csv", [dict(method=m, label=label(m)[0], recording=r, person=sd.PERSON[r],
                                     file=sd.RECORDINGS[r], macro_f1=r4(p["phase"]["macro_f1"]),
                                     n_samples=p["phase"]["n"], **ev_cols(p))
                                for m in methods for r, p in s["per_rec"][m].items()])

    curve = []
    for scheme in ("loro", "lopo"):
        base = s["methods"].get(f"ml@{scheme}")
        for src in SOURCES:
            for x in (0, *X_LIST):
                key = f"ml@{scheme}" if x == 0 else f"online_{src}_{x}@{scheme}"
                if key not in s["methods"]:
                    continue
                p = s["methods"][key]
                ci = s["ci"].get(key, {})
                curve.append(dict(scheme=scheme, source=src, source_label=SRC_LABEL[src], x_s=x,
                                  macro_f1=r4(p["phase"]["macro_f1"]), macro_f1_lo=r4(ci.get("macro_f1", [None])[0]),
                                  macro_f1_hi=r4(ci.get("macro_f1", [None, None])[1]),
                                  HS_f1=r4(p["events"]["HS"]["f1"]), HS_recall=r4(p["events"]["HS"]["recall"]),
                                  HS_fa_per_100=r4(p["events"]["HS"]["fa_per_100"]),
                                  HS_err_median_ms=r1(p["events"]["HS"]["err_median_ms"]),
                                  TO_recall=r4(p["events"]["TO"]["recall"]),
                                  delta_macro_f1_vs_base=r4(p["phase"]["macro_f1"] - base["phase"]["macro_f1"]) if base else None))
    write("online_curve.csv", curve)

    write("confusion.csv", [dict(method=m, label=label(m)[0], true_phase=PHASES[i], predicted_phase=PHASES[j],
                                 count=s["methods"][m]["cm"][i][j],
                                 row_share=r4(s["methods"][m]["cm"][i][j] / max(1, sum(s["methods"][m]["cm"][i]))))
                            for m in methods if "online" not in m for i in range(4) for j in range(4)])

    write("paired.csv", [dict(comparison=k, metric=mt, lo=r4(v[0]), median=r4(v[1]), hi=r4(v[2]))
                         for k, d in s["paired_diff"].items() for mt, v in d.items()
                         if k.startswith(("fsm", "ml@")) or "fsm" in k or "ml@" in k])

    ag = []
    for comp, d in s["agreement"].items():
        for k in TYPES:
            e = d["events"][k]
            ag.append(dict(comparison=comp, mark=k, chunks=len(d["chunks"]), n_first=e["n_a"], n_second=e["n_b"],
                           matched=e["matched"], abs_dt_median_ms=e["abs_dt_median_ms"], abs_dt_p90_ms=e["abs_dt_p90_ms"],
                           phase_sample_agreement=r4(d["phase_agreement"])))
    write("agreement.csv", ag)

    write("rule_vs_manual.csv", [dict(mark=k, **v) for k, v in s["rule_vs_manual"].items()])

    write("test_set.csv", [dict(recording=r, file=sd.RECORDINGS[r], person=sd.PERSON[r],
                                **{k: v for k, v in s["test_set"][r].items()},
                                fsm_swingEnter=s["fsm_thresholds"][r]["swingEnter"],
                                fsm_pushOffEnter=s["fsm_thresholds"][r]["pushOffEnter"],
                                fsm_heelStrike=s["fsm_thresholds"][r]["heelStrike"])
                           for r in s["test_set"]])

    write("folds.csv", [dict(scheme=f["scheme"], test=f["test"], val=f["val"], config=f["cfg"], debounce=f["debounce"],
                             n_params=f["n_params"], macs=f["macs"]) for f in s["folds"]])
