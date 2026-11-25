"""Turns the manual (by-eye) marks into one marks file per recording and into
per-sample phase labels.

A phase runs from one mark to the next one in the cycle
HS -> PO -> TO -> MS -> HS:  HS..PO = STANCE, PO..TO = PRE_SWING,
TO..MS = SWING, MS..HS = TERMINAL_SWING. Standing-still spans are STANCE.
Anything else (out-of-order marks, very long gaps, 'unclear' spans) is -1,
i.e. left out of training and scoring.
Usage: labels.py merge [final|reviewed|raw-A|raw-B]  -> annotations/merged/<REC>_marks.json
(final = reviewed file where the chunk was reviewed, otherwise the annotator's file)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saker_data as sd  # noqa: E402

TYPES = ["HS", "PO", "TO", "MS"]
NEXT = {"HS": "PO", "PO": "TO", "TO": "MS", "MS": "HS"}
PHASE_OF = {"HS": 0, "PO": 1, "TO": 2, "MS": 3}      # GaitPhase order on the board
PHASES = ["STANCE", "PRE_SWING", "SWING", "TERMINAL_SWING"]
MAX_S = {"HS": 2.5, "PO": 1.0, "TO": 1.0, "MS": 1.2}  # longest believable span from each mark
DEDUP_S = 0.08


def chunk_files(kind):
    """kind 'final' -> the reviewed file where a chunk was reviewed, else the annotator's;
    'reviewed' -> annotations/reviewed/<id>.json only; 'raw-A'/'raw-B' -> annotations/raw/<id>_A.json."""
    with open(sd.ANNOT / "chunks.json") as f:
        chunks = json.load(f)
    out = []
    for c in chunks:
        rev = sd.ANNOT / "reviewed" / f"{c['id']}.json"
        raw_a = sd.ANNOT / "raw" / f"{c['id']}_A.json"
        if kind == "final":
            p = rev if rev.exists() else raw_a
        elif kind == "reviewed":
            p = rev
        else:
            p = sd.ANNOT / "raw" / f"{c['id']}_{kind[-1]}.json"
        if p.exists():
            out.append((c, p))
    return out


def _clip_spans(spans, a, b):
    return [[max(x, a), min(y, b)] for x, y in spans if y > a and x < b and min(y, b) > max(x, a)]


def _union(spans):
    out = []
    for x, y in sorted(spans):
        if out and x <= out[-1][1]:
            out[-1][1] = max(out[-1][1], y)
        else:
            out.append([x, y])
    return out


def merge(kind="final"):
    per_rec = {}
    for c, p in chunk_files(kind):
        with open(p) as f:
            m = json.load(f)
        r = per_rec.setdefault(c["rec"], {"rec": c["rec"], "events": [], "still": [], "unclear": [], "chunks": []})
        r["chunks"].append(c["id"])
        r["events"] += [dict(e, chunk=c["id"]) for e in m.get("events", [])
                        if e.get("type") in TYPES and c["start"] <= e["t"] < c["end"]]
        r["still"] += _clip_spans(m.get("still", []), c["start"], c["end"])
        r["unclear"] += _clip_spans(m.get("unclear", []), c["start"], c["end"])
    for r in per_rec.values():
        ev = sorted(r["events"], key=lambda e: e["t"])
        kept = []
        for e in ev:   # the same landmark marked from both sides of a chunk edge
            if kept and kept[-1]["type"] == e["type"] and e["t"] - kept[-1]["t"] < DEDUP_S \
                    and kept[-1]["chunk"] != e["chunk"]:
                kept[-1]["t"] = round((kept[-1]["t"] + e["t"]) / 2, 3)
                continue
            kept.append(e)
        r["events"] = kept
        r["still"] = _union(r["still"])
        r["unclear"] = _union(r["unclear"])
    return per_rec


def phase_labels(marks, tg):
    """Per-sample phase (0..3, -1 = unlabelled) on times tg (seconds)."""
    lab = np.full(len(tg), -1, int)
    ev = sorted(marks["events"], key=lambda e: e["t"])
    still = marks.get("still", [])
    for a, b in zip(ev, ev[1:]):
        if NEXT[a["type"]] != b["type"]:
            continue
        long_ok = a["type"] == "HS" and any(x < b["t"] and y > a["t"] for x, y in still)
        if b["t"] - a["t"] > MAX_S[a["type"]] and not long_ok:
            continue
        lab[(tg >= a["t"]) & (tg < b["t"])] = PHASE_OF[a["type"]]
    for x, y in still:
        lab[(tg >= x) & (tg < y)] = 0
    for x, y in marks.get("unclear", []):
        lab[(tg >= x) & (tg < y)] = -1
    return lab


def load_marks(rec, kind="merged"):
    with open(sd.ANNOT / kind / f"{rec}_marks.json") as f:
        return json.load(f)


if __name__ == "__main__":
    kind = sys.argv[2] if len(sys.argv) > 2 else "final"
    if sys.argv[1:2] == ["merge"]:
        out_dir = sd.ANNOT / ("merged" if kind == "final" else f"merged_{kind}")
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec, r in merge(kind).items():
            with open(out_dir / f"{rec}_marks.json", "w") as f:
                json.dump(r, f, indent=1)
            counts = {k: sum(e["type"] == k for e in r["events"]) for k in TYPES}
            print(rec, len(r["chunks"]), "chunks", counts, "still", len(r["still"]), "unclear", len(r["unclear"]))
    else:
        print(__doc__)
