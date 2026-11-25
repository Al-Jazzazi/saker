"""Splits every recording into ~1-minute annotation chunks and pre-renders the
5 s overview windows each annotator starts from.
Usage: plan_chunks.py IMG_DIR  ->  writes annotations/chunks.json and the PNGs."""
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annot_tool  # noqa: E402
import saker_data as sd  # noqa: E402

CHUNK_S, WIN_S, STEP_S, PAD_S = 60.0, 5.0, 4.0, 0.5
FINE_S, FINE_STEP_S, FINE_PAD_S = 3.0, 2.5, 0.25   # fine-grid windows for placing marks


def plan(img_dir):
    chunks = []
    for rec in sd.RECORDINGS:
        d = sd.load(rec)
        dur = float(d["t"][-1])
        n = math.ceil(dur / CHUNK_S)
        edges = [round(dur * i / n, 2) for i in range(n + 1)]
        edges[-1] = round(dur + 0.01, 2)
        segs = sd.task_segments(d)
        for i in range(n):
            s, e = edges[i], edges[i + 1]
            wins = []
            w0 = s - PAD_S
            while w0 < e - PAD_S:
                ws, we = round(w0, 2), round(w0 + WIN_S, 2)
                wins.append({"start": ws, "end": we, "png": str(Path(img_dir) / f"{rec}_c{i:02d}_w{len(wins):02d}.png")})
                w0 += STEP_S
            fine = []
            w0 = s - FINE_PAD_S
            while w0 < e - FINE_PAD_S:
                ws, we = round(w0, 2), round(w0 + FINE_S, 2)
                fine.append({"start": ws, "end": we, "png": str(Path(img_dir) / f"{rec}_c{i:02d}_f{len(fine):02d}.png")})
                w0 += FINE_STEP_S
            tasks = [f"{k} {max(a, s):.1f}-{min(b, e):.1f}" for k, a, b in segs if b > s and a < e]
            chunks.append({"id": f"{rec}_c{i:02d}", "rec": rec, "start": s, "end": e, "tasks": tasks, "windows": wins,
                           "fine": fine})
    return chunks


def _render(job):
    annot_tool.render(*job)


if __name__ == "__main__":
    img = Path(sys.argv[1])
    img.mkdir(parents=True, exist_ok=True)
    chunks = plan(img)
    sd.ANNOT.mkdir(exist_ok=True)
    with open(sd.ANNOT / "chunks.json", "w") as f:
        json.dump(chunks, f, indent=1)
    jobs = [(c["rec"], w["start"], w["end"], w["png"]) for c in chunks for w in c["windows"] + c["fine"]
            if not Path(w["png"]).exists()]
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(_render, jobs, chunksize=8))
    print(len(chunks), "chunks,", len(jobs), "windows rendered")
