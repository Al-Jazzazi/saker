#pragma once
// Gait-phase state machine for the healthy leg, plus the WALK calibration that
// tunes it. Plain C++ with no Arduino calls, so tests/test_gait_fsm.cpp can
// compile and test it on the laptop.
//
// Input is the shank's sagittal angular velocity ("shank rate", deg/s), signed
// so it is positive while the shank swings forward. One stride looks like:
//
//   heel strike   stance        push-off   toe-off   mid-swing      heel strike
//        |  dip  ~0 / slightly -   big dip    |    large + peak  falls |
//        v                                    v                        v
//   STANCE -----------------> PRE_SWING ---> SWING ---> TERMINAL_SWING ---> STANCE
//
// Every transition is a rule on the shank rate plus a minimum/maximum time in
// the phase. Thresholds come from the WALK calibration; timings live in
// GaitTiming so rule variants can be tried without touching the logic.
#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <algorithm>

enum GaitPhase : uint8_t { PHASE_STANCE, PHASE_PRE_SWING, PHASE_SWING, PHASE_TERMINAL_SWING };

// What happened on a sample. TIMEOUT means a phase ran too long and the
// machine fell back to STANCE; it never triggers the motor.
enum GaitEvent : uint8_t {
  EVENT_NONE, EVENT_PUSH_OFF, EVENT_TOE_OFF, EVENT_MID_SWING, EVENT_HEEL_STRIKE, EVENT_TIMEOUT
};

inline const char *gaitPhaseName(GaitPhase phase) {
  switch (phase) {
    case PHASE_STANCE: return "STANCE";
    case PHASE_PRE_SWING: return "PRE_SWING";
    case PHASE_SWING: return "SWING";
    case PHASE_TERMINAL_SWING: return "TERMINAL_SWING";
  }
  return "?";
}

inline const char *gaitEventName(GaitEvent event) {
  switch (event) {
    case EVENT_NONE: return "";
    case EVENT_PUSH_OFF: return "PUSH_OFF";
    case EVENT_TOE_OFF: return "TOE_OFF";
    case EVENT_MID_SWING: return "MID_SWING";
    case EVENT_HEEL_STRIKE: return "HEEL_STRIKE";
    case EVENT_TIMEOUT: return "TIMEOUT";
  }
  return "?";
}

// Shank-rate levels in deg/s. Defaults are only placeholders: WALK replaces them.
struct GaitThresholds {
  float swingEnter = 100;    // above this (with the thigh moving) the foot is off the ground
  float pushOffEnter = -100; // below this, after the minimum stance time, push-off has begun
  float heelStrike = -15;    // below this in terminal swing, the heel has landed
};

struct GaitTiming {
  uint32_t minStanceMs = 200;         // skip the heel-strike dip at the start of stance
  uint32_t maxPreSwingMs = 500;       // push-off that never becomes swing was a false alarm
  uint32_t minSwingMs = 80;           // don't look for the mid-swing peak before this
  uint32_t maxSwingMs = 800;
  uint32_t maxTerminalSwingMs = 400;
  float midSwingDropRatio = 0.8f;     // peak has passed once the rate drops below 80% of its max
  float thighMovingDps = 10;          // thigh must also move, so a shank-only twitch is not a step
};

class GaitFsm {
 public:
  GaitThresholds thresholds;
  GaitTiming timing;

  void reset(uint32_t nowMs) {
    phase_ = PHASE_STANCE;
    enteredMs_ = nowMs;
    swingMax_ = 0;
  }

  GaitPhase phase() const { return phase_; }

  // Feed one sample. Returns the event that happened on it (usually EVENT_NONE).
  GaitEvent update(float shankRate, float thighRateMagnitude, uint32_t nowMs) {
    const uint32_t inPhaseMs = nowMs - enteredMs_;
    const bool thighMoving = thighRateMagnitude > timing.thighMovingDps;

    switch (phase_) {
      case PHASE_STANCE:
        if (inPhaseMs < timing.minStanceMs) break;
        // Slow walking can skip a clear push-off, so allow stance -> swing directly.
        if (shankRate > thresholds.swingEnter && thighMoving) return enterSwing(nowMs);
        if (shankRate < thresholds.pushOffEnter) return enter(PHASE_PRE_SWING, EVENT_PUSH_OFF, nowMs);
        break;

      case PHASE_PRE_SWING:
        if (shankRate > thresholds.swingEnter && thighMoving) return enterSwing(nowMs);
        if (inPhaseMs > timing.maxPreSwingMs) return enter(PHASE_STANCE, EVENT_TIMEOUT, nowMs);
        break;

      case PHASE_SWING:
        swingMax_ = std::max(swingMax_, shankRate);
        if (inPhaseMs >= timing.minSwingMs && shankRate < swingMax_ * timing.midSwingDropRatio) {
          return enter(PHASE_TERMINAL_SWING, EVENT_MID_SWING, nowMs);
        }
        if (inPhaseMs > timing.maxSwingMs) return enter(PHASE_STANCE, EVENT_TIMEOUT, nowMs);
        break;

      case PHASE_TERMINAL_SWING:
        if (shankRate < thresholds.heelStrike) return enter(PHASE_STANCE, EVENT_HEEL_STRIKE, nowMs);
        if (inPhaseMs > timing.maxTerminalSwingMs) return enter(PHASE_STANCE, EVENT_TIMEOUT, nowMs);
        break;
    }
    return EVENT_NONE;
  }

 private:
  GaitPhase phase_ = PHASE_STANCE;
  uint32_t enteredMs_ = 0;
  float swingMax_ = 0;

  GaitEvent enter(GaitPhase next, GaitEvent event, uint32_t nowMs) {
    phase_ = next;
    enteredMs_ = nowMs;
    return event;
  }

  GaitEvent enterSwing(uint32_t nowMs) {
    swingMax_ = 0;
    return enter(PHASE_SWING, EVENT_TOE_OFF, nowMs);
  }
};

// ---------------------------------------------------------------------------
// WALK calibration: from ~12 s of normal walking, find which gyro axis is the
// sagittal one on each sensor (so mounting orientation doesn't matter), which
// sign means "swinging forward", and the shank-rate thresholds for the FSM.

struct SagittalAxis {
  uint8_t axis = 0;  // 0 = x, 1 = y, 2 = z
  int8_t sign = 1;
};

struct WalkCalResult {
  bool ok = false;
  const char *message = "";
  SagittalAxis shank, thigh;
  float swingPeak = 0;    // typical mid-swing shank rate, deg/s
  float pushOffPeak = 0;  // typical size of the negative push-off / heel-strike dips, deg/s
  int steps = 0;
  GaitThresholds thresholds;
};

namespace walkcal {

constexpr float MIN_SWING_PEAK_DPS = 60;  // below this the person was not really walking
constexpr uint32_t MIN_STEP_GAP_MS = 300;

// Value at quantile q (0..1) of n values; reorders `values`.
inline float quantile(float *values, size_t n, float q) {
  const size_t k = std::min(n - 1, (size_t)(q * (n - 1) + 0.5f));
  std::nth_element(values, values + k, values + n);
  return values[k];
}

// The sagittal axis is the one that moves most while walking.
inline uint8_t mostActiveAxis(const float (*gyro)[3], size_t n) {
  double sum[3] = {}, sumSq[3] = {};
  for (size_t i = 0; i < n; i++) {
    for (int a = 0; a < 3; a++) {
      sum[a] += gyro[i][a];
      sumSq[a] += (double)gyro[i][a] * gyro[i][a];
    }
  }
  uint8_t best = 0;
  double bestVar = -1;
  for (uint8_t a = 0; a < 3; a++) {
    const double var = sumSq[a] / n - (sum[a] / n) * (sum[a] / n);
    if (var > bestVar) { bestVar = var; best = a; }
  }
  return best;
}

}  // namespace walkcal

// shankGyro / thighGyro: n bias-corrected samples (deg/s) taken every sampleMs.
// scratch: room for n floats.
inline WalkCalResult analyseWalk(const float (*shankGyro)[3], const float (*thighGyro)[3],
                                 size_t n, uint32_t sampleMs, float *scratch) {
  using namespace walkcal;
  WalkCalResult r;
  if (n < 2000 / sampleMs) { r.message = "too few samples"; return r; }

  // Shank: the forward swing is the largest peak of the stride, so the sign is
  // whichever side of the sagittal axis reaches further.
  r.shank.axis = mostActiveAxis(shankGyro, n);
  for (size_t i = 0; i < n; i++) scratch[i] = shankGyro[i][r.shank.axis];
  const float high = quantile(scratch, n, 0.98f);
  const float low = quantile(scratch, n, 0.02f);
  r.shank.sign = (high >= -low) ? 1 : -1;
  r.swingPeak = (r.shank.sign > 0) ? high : -low;
  r.pushOffPeak = (r.shank.sign > 0) ? -low : high;

  // Thigh: choose the sign that moves together with the shank (both swing forward).
  r.thigh.axis = mostActiveAxis(thighGyro, n);
  double together = 0;
  for (size_t i = 0; i < n; i++) {
    together += (double)thighGyro[i][r.thigh.axis] * r.shank.sign * shankGyro[i][r.shank.axis];
  }
  r.thigh.sign = (together >= 0) ? 1 : -1;

  r.thresholds.swingEnter = std::min(std::max(0.30f * r.swingPeak, 30.0f), 200.0f);
  r.thresholds.pushOffEnter = -std::min(std::max(0.60f * r.pushOffPeak, 20.0f), 200.0f);
  r.thresholds.heelStrike = -std::min(std::max(0.10f * r.pushOffPeak, 5.0f), 40.0f);

  // Count strides: upward crossings of the swing threshold, at least MIN_STEP_GAP_MS apart.
  bool above = false;
  size_t lastStep = 0;
  for (size_t i = 0; i < n; i++) {
    const float rate = r.shank.sign * shankGyro[i][r.shank.axis];
    if (!above && rate > r.thresholds.swingEnter) {
      if (r.steps == 0 || (i - lastStep) * sampleMs >= MIN_STEP_GAP_MS) {
        r.steps++;
        lastStep = i;
      }
      above = true;
    } else if (above && rate < 0.5f * r.thresholds.swingEnter) {
      above = false;
    }
  }

  if (r.swingPeak < MIN_SWING_PEAK_DPS) { r.message = "barely any leg swing seen"; return r; }
  if (r.steps < 4) { r.message = "fewer than 4 steps seen"; return r; }
  r.ok = true;
  r.message = "ok";
  return r;
}
