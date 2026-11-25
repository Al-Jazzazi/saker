## 8. Interpretation (written from the final run; numbers are in tables 1–3)

### FSM (board GaitFsm, 12 s calibration)

- **Heel strike, the only event that fires the motor, is excellent.**
  - It finds 98.7% of heel strikes, with 1.2 false ones per 100 strides and only 2 during standing.
  - It fires a median 30 ms early: it triggers when the shank rate drops below `heelStrike` just after the zero crossing, slightly before the impact. The P90 error is 60 ms.
  - Its HS F1 (0.987) is the best of every method tested.
- **The phases are poor (macro F1 0.59)** because its thresholds fire late by design:
  - toe-off is a median 140 ms late (it waits for the shank to rise past swingEnter), so only 27% land within ±150 ms, though 99% land within ±300 ms;
  - mid-swing is 60 ms late (it waits for the drop to 80% of the peak);
  - push-off is 90–200 ms late (it waits for 60% of the dip).
- So the FSM is late, not lost. It cuts the stride at the wrong moments, but it still produces one clean stride per stride.

### ML (MLP, 3k weights, trained on other recordings)

- **The phases are far better (macro F1 0.88).** Every event is a median 10 ms from the manual mark, and toe-off and mid-swing are found ≥ 99.8% of the time.
- **It generalises to a new person**, at least with 2 people: macro F1 drops only from 0.879 (new recording) to 0.869 (new person).
- **But it fires many false heel strikes: about 32–37 per 100 strides.** The phase output flickers around stance/terminal swing, and each flicker becomes an extra heel-strike event. Trained on the other person, it also gave 27 false heel strikes while standing.
- As a motor trigger it's worse than the FSM. That would need a refractory period or an FSM-style "one heel strike per stride" gate on top.

### ML + online learning (fine-tune on the first x s of normal walking)

- **With manual labels it helps, and it helps most on a new person.**
  - Fine-tuning on 30 s raises macro F1 to 0.893–0.895.
  - False heel strikes fall from 32 to 7.5 per 100 strides (new person) and from 37 to 20.5 (new recording).
  - Even 12 s gives most of the gain.
  - But someone has to label those seconds, so this is an upper bound, not something the board can do by itself.
- **The two label sources the board could use by itself don't help:**
  - **FSM pseudo-labels make it much worse** (macro F1 about 0.58–0.64). The model learns the FSM's lag: its toe-offs and push-offs move late like the FSM's.
  - **Look-ahead pseudo-labels** (the offline shank-gyro rule run on the buffer) **are slightly worse than no fine-tuning** (0.82–0.84 vs 0.87–0.88). That rule puts push-off at "12% of the stride before toe-off", which disagrees with where push-off actually is (median +110 ms vs the manual marks), so the model gets pulled away from the truth.
- **Conclusion: online learning is only worth it if the labels for the fine-tuning seconds are good.** Self-labelling with the FSM or the simple rule doesn't buy anything here.

### Different models and sensors (table 1b)

- **Tree ensembles are the best ML models.** All six models get the same features and the same test set.
  - Random forest: macro F1 0.901 on both a new recording and a new person.
  - Gradient boosting: 0.897 on a new recording, 0.863 on a new person.
  - Both beat the MLP (0.879 / 0.869), and the linear models trail (0.84–0.85).
  - Gradient boosting has the fewest false heel strikes of any ML model: 17 per 100 strides, 9.4 with the gate on a new person.
- **The shank carries the signal.**
  - An MLP on the shank IMU alone (0.894) does as well as both IMUs.
  - The thigh alone is poor (0.63–0.70), with many false heel strikes.
  - This matches the FSM's design, which also works mainly from the shank.
- **The "one heel strike per stride" gate** only lets phases move forward round the cycle, and requires a swing and ≥ 300 ms between heel strikes.
  - It halves false heel strikes: MLP 36.6 → 18.3, gradient boosting 17.0 → 12.8.
  - It costs 0.01–0.02 macro F1 and, for some models, some recall.
  - Even the best gated ML (9.4 per 100) is still about 8× noisier than the FSM (1.2 per 100).

### What to use

- **Motor trigger (heel strike): keep the FSM.** It has the most reliable heel strike with the fewest false triggers. Firing about 30 ms early is harmless, and arguably helpful, given the motor's 1.5 s move.
- **If you need the phases themselves** (assistance timed inside swing, or swing/stance percentages): use the ML model, but take heel strike from the FSM, or gate the ML heel strikes (refractory period ≥ 300 ms, only allowed after a swing).
- **Online learning** is worth revisiting only with a better self-labeller, or a short labelled calibration walk per user. With FSM labels it actively hurts.

### Caveats specific to these numbers

- **Ground-truth quality was measured less than planned.** One chunk got a full independent review (155 marks, none moved more than 15 ms). Three more reviews were stopped part-way to save credits, so they show little change, and the blind double annotations were cancelled.
- **The by-eye marks were checked against the textbook shank-gyro rule:** toe-off and mid-swing match it with median 0 ms and P90 10 ms, and heel strike with median −10 ms and P90 70 ms (the manual heel strike is the impact onset, the rule's is the dip). The clear landmarks therefore agree with the standard definition. Push-off has no rule equivalent and is the softest mark: most push-off marks are flagged unsure, and only sure marks are scored as events.
- **The last 14 chunks were completed by Sonnet at low effort.** Most of their marks came from earlier Opus attempts, and Sonnet only spot-checked them.
