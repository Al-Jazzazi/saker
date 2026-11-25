# SAKER manual gait annotation guide

Every mark is placed **by eye** on plots of the IMU data, one stride at a time. No code finds the events: no peak finding, no thresholds, no reading numbers out of the CSV. These marks are the ground truth that the board's FSM, an ML model and an ML model with online learning get scored against, so be careful and honest. "unsure" and "unclear" are always better than a guess.

## What you see in a plot

`annot_tool.py render` draws three stacked panels sharing one time axis (seconds since the recording started):

| panel | signal | sign |
|---|---|---|
| top (big) | **shank** sagittal angular rate, deg/s | + = shank swinging forward (swing) |
| middle | **thigh** sagittal rate, deg/s | helper only; it is noisy for some people |
| bottom | shank acceleration magnitude − 1 g | sharp spikes = impacts (foot contact) |

- Grid in 5 s windows: light lines every 0.1 s, dark lines and labels every 0.5 s. Windows of 3 s or less get a fine grid: light lines every 0.02 s, dark lines and labels every 0.1 s. Use those zooms to place marks to ±10 ms.
- Dots are the real 100 Hz samples.
- **Red bands are Wi-Fi gaps.** No samples exist there; the line is broken. Place an event inside a gap only if the trace on both sides makes its position obvious (±20 ms), and mark it `"conf": "unsure"`. If you can't tell, leave that mark out and cover the stretch with an `unclear` span.

## A normal stride, in order

`... HS → (stance) → PO → TO → (swing) → MS → HS ...`

The board's FSM cycles STANCE → PRE_SWING → SWING → TERMINAL_SWING. Each mark starts one phase:

| mark | starts | where exactly |
|---|---|---|
| **HS** heel strike (foot contact) | STANCE | **The start of the first sharp impact spike in the bottom panel after the swing peak**: the last sample before the |acc| line shoots up. In the top panel, the shank has just fallen through zero and shows a jolt (sharp dip, kink or bounce) at the same moment. Careful: in slippers (Amjad) the shank often makes a smooth negative dip 30–100 ms **before** the impact. That dip is not the contact; the impact is. If there is no clear impact spike (soft landing, gap), use the bottom of the first sharp negative shank dip after the zero crossing and mark it `unsure`. |
| **PO** push-off (pre-swing start) | PRE_SWING | Late stance. The shank sits at a fairly flat negative level (a plateau or gentle hump) and then **starts its continuous fall into the big toe-off dip**. PO is that bend: the last point of the plateau/hump, after which the trace keeps going down all the way to TO. If the stance part has small bumps, PO is the top of the last bump before the fall. It's usually 150–400 ms before TO. It's the softest landmark, so zoom in and use `unsure` freely. |
| **TO** toe-off | SWING | **The bottom (most negative point) of the big negative dip** right before the shank's steep rise into swing. If the bottom is flat, take its middle. |
| **MS** mid-swing | TERMINAL_SWING | **The top of the swing peak** (the largest positive shank rate of that stride). If the top is flat or has two humps, take the middle of the top. |

Ignore post-impact ringing: after HS the shank and thigh often bounce for 100–300 ms. That's stance; don't mark anything there.

## Not normal walking

- **still** span: the leg is standing still (traces flat near zero, tiny wobble). It counts as STANCE. Typical: the stand tasks and the pauses in stop_go. Begin it after the last HS settles and end it where the leg starts moving again.
- **irregular** span: turning around, pivoting, shuffling or tiny steps. Inside it, still mark every landmark you can actually see (use `unsure`), and put `unclear` over the parts you can't read. Each walk task has turnarounds (look for small swing peaks under ~120 deg/s, missing impacts, odd bumps).
- **unclear** span: you can't tell what the leg is doing, or a gap hides a mark. Everything inside is left out of scoring.
- stop_go: the first stride after standing often has a small odd PO/TO. Mark what you see. Where the leg stops, the last mark is usually HS, followed later by a `still` span.
- **Never invent a stride or copy a neighbour's spacing.** Look at each stride on its own.

## Worked examples (marks drawn on top)

- `examples/example_YL3.png` with `examples/example_YL3.json`: normal walking and a turnaround (purple bar = irregular, hatched = unclear, dashed = unsure).
- `examples/example_AL2.png` with `examples/example_AL2.json`: Amjad in slippers. The shank dips at 399.13 but the impact starts at 399.20, so HS = 399.20. PO is where the plateau ends at 399.77, and TO is the middle of the flat dip bottom at 400.05.

## File format (JSON)

```json
{
  "rec": "YL3", "start": 56.25, "end": 112.5, "annotator": "A",
  "events": [
    {"t": 61.43, "type": "HS"},
    {"t": 61.96, "type": "PO", "conf": "unsure"},
    {"t": 62.24, "type": "TO"},
    {"t": 62.44, "type": "MS"}
  ],
  "still": [[130.1, 139.6]],
  "unclear": [[101.2, 102.9]],
  "irregular": [[100.5, 104.0]],
  "notes": "short free text"
}
```

- `t` is in seconds as read off the time axis, to 0.01 s. Add `"conf": "unsure"` when you could be more than ~40 ms off or aren't sure the landmark is real.
- Only events with `start <= t < end` belong to the chunk. The windows overhang the edges for context. Spans may overhang; they get clipped.
- Order must be HS → PO → TO → MS → HS. If a mark is missing on purpose, cover that stretch with an `unclear` (or `still`) span so `check` knows it's deliberate.
