# Handoff: SAKER gait-detector benchmark

For whoever picks this up (Codex or a person). Everything below runs locally; no AI agents are needed from here on.

## The task (agreed with Yousef)

1. Annotate every stride in the 7 real recordings **by eye** (no rule-based labelling), with the FSM's phases:
   HS (heel strike) → STANCE, PO (push-off) → PRE_SWING, TO (toe-off) → SWING, MS (mid-swing) → TERMINAL_SWING.
2. Compare three detectors **on the identical test set**:
   - **FSM**: the board's `GaitFsm.h`, calibrated by `analyseWalk` on the first 12 s of walk_normal of each recording.
   - **ML**: a model trained on other recordings (leave-one-recording-out, plus leave-one-person-out).
   - **ML + online learning**: the same model, fine-tuned on the first x s (6/12/20/30) of the test recording's walk_normal.
3. For each method: analysis (numbers) and interpretation (what it means for firing the motor at heel strike).

**Same-test-set rule.** In each recording the setup window (opening stand + first 30 s of walk_normal) is never scored. Everything after `walk0 + 30 s` is the test set for every method.

## State

- **Manual marks:** `../annotations/raw/<chunk>_A.json`, 50 chunks of about 1 minute (`../annotations/chunks.json`).
  - 4 chunks were independently reviewed: `../annotations/reviewed/<chunk>.json`. Where a reviewed file exists it wins (`labels.py merge` handles this).
  - The definitions annotators followed are in `../annotations/ANNOTATION_GUIDE.md`, with worked examples in `../annotations/examples/`.
  - 36 chunks were marked by Opus agents; the last 14 by Sonnet at low effort (to save credits): AL2_c00, c01, c03, c04, c05, c06, c07; YL1_c01, c02; YL3_c06, c07; YR2_c05, c06, c07.
  - If a chunk's `_A.json` fails `annot_tool.py check`, fix it by eye with the tools below, or cover the bad stretch with an `"unclear"` span.
- **Pipeline:** written and tested end to end. One command:
  ```sh
  cd laptop/analysis
  ./run_all.sh /path/to/python   # needs numpy scipy scikit-learn matplotlib; ~10-30 min on 8 cores
  ```
  - Outputs: `../results/benchmark/REPORT.md` (tables, figures, auto-filled headline), `summary.json`, and CSVs in `report/`.

## Files (`laptop/analysis/`)

| file | what it does |
|---|---|
| `saker_data.py` | loads a recording. Bias from the opening stand; shank sagittal axis/sign from the board's WALK calibration (`results/walk_cal_thresholds.csv`); thigh = main rotation direction; 100 Hz grid with Wi-Fi gaps filled |
| `annot_tool.py` | plot viewer for by-eye annotation (`render`, `overlay`, `check`); draws only, never detects |
| `plan_chunks.py` | chunk list + pre-rendered plot windows (images in a scratch folder; re-run to regenerate) |
| `labels.py` | merges chunk marks per recording → per-sample phase labels (−1 = unlabelled) |
| `fsm_replay.cpp`, `walk_cal.cpp` | the board's own GaitFsm / analyseWalk compiled for the laptop |
| `methods.py` | FSM runner, causal features (past ≤ 1 s), numpy MLP with fine-tuning, debounce, phase→event |
| `scoring.py` | phase confusion/F1; events matched one-to-one within ±150 ms (also ±300 ms); false alarms; timing |
| `benchmark.py` | runs all methods/folds → `results/benchmark/raw_results.npz` |
| `analyze.py` | pooled / per-task / per-person / per-recording scores, block-bootstrap CIs, annotator agreement |
| `export_report.py` | CSVs for the report page |
| `write_report.py` | `REPORT.md` + `figures/` |

## What's left

Done as of 2026-11-25: all 50 chunks annotated, final pipeline run, `REPORT.md` written, and the interpretation is in `results/benchmark/INTERPRETATION.md` (included in REPORT.md).

Optional next steps:
- Gate ML heel strikes (refractory period ≥ 300 ms, only allowed after a swing) and re-score. This is the obvious fix for ML's ~35 false heel strikes per 100 strides.
- Re-run the blind double annotation on 3–5 chunks to measure ground-truth agreement properly.
- Publish the CSVs in `results/benchmark/report/` as a page.

## Rules that still apply

- Don't upload to the ESP32 or hold its serial port; Yousef uploads from the Arduino IDE.
- `SAKER_Sender/secrets.h` (Wi-Fi password) must never be committed; the GitHub repo is public.
- Commits: no co-author or "generated with" trailers.
