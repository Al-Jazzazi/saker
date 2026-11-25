"""Plot viewer for manual gait annotation. It only draws pictures and checks the
shape of a marks file; it never finds events itself.

  render REC START END OUT.png [--marks FILE.json]
      Shank sagittal rate (top, + = shank swinging forward), thigh sagittal rate
      (middle) and shank |acc| - 1 g (bottom) for START..END seconds. Red bands are
      Wi-Fi gaps (no samples). Windows up to 3 s get a fine 20 ms grid.
      --marks draws the marks in FILE on top (HS red, PO orange, TO blue,
      MS green; dashed = unsure), shades the phases between them, hatches
      unclear spans and draws irregular spans as a purple bar along the top.
  overlay CHUNK_ID FILE.json OUT_PREFIX [fine|coarse]
      Renders FILE's marks over every window of a chunk in one go: fine = the
      3 s windows with the 20 ms grid (default), coarse = the 5 s windows.
      Prints the PNG paths.
  check FILE.json
      Lists ordering problems in a marks file (expected HS -> PO -> TO -> MS -> HS).
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import saker_data as sd  # noqa: E402

TYPES = ["HS", "PO", "TO", "MS"]
NEXT = {"HS": "PO", "PO": "TO", "TO": "MS", "MS": "HS"}
COLOR = {"HS": "#d62728", "PO": "#ff7f0e", "TO": "#1f77b4", "MS": "#2ca02c"}
PHASE_OF = {"HS": "STANCE", "PO": "PRE_SWING", "TO": "SWING", "MS": "TERMINAL_SWING"}
PHASE_COLOR = {"STANCE": "#bbbbbb", "PRE_SWING": "#ffcc88", "SWING": "#9ecae1", "TERMINAL_SWING": "#a1d99b"}
MAX_PHASE_S = 2.0   # longer than this between two marks is drawn as unknown
_load = lru_cache(maxsize=2)(sd.load)


def _load_marks(path):
    with open(path) as f:
        m = json.load(f)
    m.setdefault("events", [])
    m.setdefault("still", [])
    m.setdefault("unclear", [])
    m.setdefault("irregular", [])
    return m


def render(rec, start, end, out, marks=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    d = _load(rec)
    t = d["t"]
    sel = (t >= start - 0.05) & (t <= end + 0.05)
    fine = (end - start) <= 3.0
    fig = plt.figure(figsize=(15.6, 8.4), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 1.4, 1.0], hspace=0.08,
                          left=0.05, right=0.99, top=0.91, bottom=0.05)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]
    series = [("shank_sag", "shank deg/s", "k"), ("thigh_sag", "thigh deg/s", "#1a4f8a"),
              ("shank_acc_dyn", "shank |a|-1 g", "#8a1a1a")]
    gaps = [(a, b) for a, b in sd.gaps(d) if b > start and a < end]
    for ax, (key, lab, c) in zip(axes, series):
        y = d[key][sel]
        x = t[sel]
        # break the line across gaps so nothing is drawn where no samples exist
        brk = np.flatnonzero(np.diff(d["gi"][sel]) > 1) + 1
        for xs, ys in zip(np.split(x, brk), np.split(y, brk)):
            ax.plot(xs, ys, color=c, lw=1.1)
        ax.plot(x, y, ".", color=c, ms=2.2 if not fine else 3.5)
        ax.axhline(0, color="k", lw=0.8)
        for a, b in gaps:
            ax.axvspan(a, b, color="#ff4444", alpha=0.22, lw=0)
        ax.set_xlim(start, end)
        ax.set_ylabel(lab, fontsize=9)
        if fine:
            ax.xaxis.set_major_locator(MultipleLocator(0.1))
            ax.xaxis.set_minor_locator(MultipleLocator(0.02))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        else:
            ax.xaxis.set_major_locator(MultipleLocator(0.5))
            ax.xaxis.set_minor_locator(MultipleLocator(0.1))
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.grid(which="major", axis="x", color="#666666", lw=0.7)
        ax.grid(which="minor", axis="x", color="#cccccc", lw=0.5)
        ax.grid(which="major", axis="y", color="#e0e0e0", lw=0.5)
        ax.tick_params(axis="x", labelsize=8, labelbottom=(ax is axes[-1]), labeltop=(ax is axes[0]))
        ax.tick_params(axis="y", labelsize=8)
    axes[0].yaxis.set_major_locator(MultipleLocator(50))

    tasks = " / ".join(dict.fromkeys(d["task"][sel])) if sel.any() else "-"
    title = f"{rec}   {start:.2f} - {end:.2f} s   task: {tasks}"
    if marks is not None:
        title += f"   marks: {Path(marks).name}"
        _draw_marks(axes, _load_marks(marks), start, end)
    fig.suptitle(title, fontsize=11, x=0.05, ha="left", y=0.985)
    fig.savefig(out)
    plt.close(fig)


def _draw_marks(axes, m, start, end):
    ev = sorted(m["events"], key=lambda e: e["t"])
    top = axes[0]
    for a, b in zip(ev, ev[1:]):
        if NEXT[a["type"]] == b["type"] and b["t"] - a["t"] <= MAX_PHASE_S:
            top.axvspan(a["t"], b["t"], color=PHASE_COLOR[PHASE_OF[a["type"]]], alpha=0.45, lw=0, zorder=0)
    for a, b in m["still"]:
        top.axvspan(a, b, facecolor="none", edgecolor="#777777", hatch="..", lw=0, zorder=0)
    for a, b in m["irregular"]:
        top.axvspan(a, b, ymin=0.97, ymax=1.0, color="#7b3294", lw=0, zorder=3)
    for a, b in m["unclear"]:
        for ax in axes:
            ax.axvspan(a, b, facecolor="none", edgecolor="#000000", hatch="///", lw=0, zorder=0)
    ylo, yhi = top.get_ylim()
    for e in ev:
        if not (start <= e["t"] <= end):
            continue
        ls = "--" if e.get("conf") == "unsure" else "-"
        for ax in axes:
            ax.axvline(e["t"], color=COLOR[e["type"]], lw=1.4, ls=ls)
        top.text(e["t"], yhi, e["type"], color=COLOR[e["type"]], fontsize=8, ha="center", va="bottom",
                 fontweight="bold", clip_on=False)


def check(path):
    m = _load_marks(path)
    problems = []
    for k in ("rec", "start", "end"):
        if k not in m:
            problems.append(f"missing key '{k}'")
    ev = m["events"]
    for e in ev:
        if e.get("type") not in TYPES or not isinstance(e.get("t"), (int, float)):
            problems.append(f"bad event {e}")
    ev = sorted([e for e in ev if e.get("type") in TYPES], key=lambda e: e["t"])
    blocked = m["still"] + m["unclear"]

    def covered(a, b):
        return any(x <= b and y >= a for x, y in blocked)

    for a, b in zip(ev, ev[1:]):
        if b["t"] - a["t"] < 0.03:
            problems.append(f"{a['type']} {a['t']:.2f} and {b['type']} {b['t']:.2f} are < 30 ms apart")
        elif NEXT[a["type"]] != b["type"] and not covered(a["t"], b["t"]):
            problems.append(f"order: {a['type']} {a['t']:.2f} then {b['type']} {b['t']:.2f} "
                            f"(expected {NEXT[a['type']]}; mark the missing one or an 'unclear' span)")
        elif b["t"] - a["t"] > MAX_PHASE_S and not covered(a["t"], b["t"]):
            problems.append(f"{b['t'] - a['t']:.2f} s from {a['type']} {a['t']:.2f} to {b['type']} {b['t']:.2f} "
                            f"with no still/unclear span")
    for name in ("still", "unclear", "irregular"):
        for span in m[name]:
            if len(span) != 2 or span[1] <= span[0]:
                problems.append(f"bad {name} span {span}")
    outside = [e for e in ev if not (m.get("start", -1e9) <= e["t"] < m.get("end", 1e9))]
    counts = {k: sum(e["type"] == k for e in ev) for k in TYPES}
    print(f"events {len(ev)}  {counts}  still spans {len(m['still'])}  unclear spans {len(m['unclear'])}")
    if outside:
        print(f"note: {len(outside)} events outside this chunk's start..end are ignored when merging")
    print("OK: no problems" if not problems else f"{len(problems)} problem(s):")
    for p in problems:
        print("  -", p)


def _render_job(job):
    render(*job)
    return job[3]


def overlay(chunk_id, marks, prefix, kind="fine"):
    from concurrent.futures import ProcessPoolExecutor
    with open(sd.ANNOT / "chunks.json") as f:
        c = next(c for c in json.load(f) if c["id"] == chunk_id)
    wins = c["fine"] if kind == "fine" else c["windows"]
    jobs = [(c["rec"], w["start"], w["end"], f"{prefix}_{kind}{i:02d}.png", marks) for i, w in enumerate(wins)]
    with ProcessPoolExecutor(3) as ex:
        for path, w in zip(ex.map(_render_job, jobs), wins):
            print(f"{path}  ({w['start']:.2f} - {w['end']:.2f} s)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "render" and len(a) >= 5:
        mk = a[a.index("--marks") + 1] if "--marks" in a else None
        render(a[1], float(a[2]), float(a[3]), a[4], mk)
        print("wrote", a[4])
    elif a and a[0] == "overlay" and len(a) in (4, 5):
        overlay(a[1], a[2], a[3], a[4] if len(a) == 5 else "fine")
    elif a and a[0] == "check" and len(a) == 2:
        check(a[1])
    else:
        print(__doc__)
        sys.exit(2)
