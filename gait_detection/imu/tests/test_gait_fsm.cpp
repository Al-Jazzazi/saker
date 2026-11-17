// Laptop test for the gait FSM and WALK calibration, using a synthetic stride.
// Build and run from the SAKER folder:
//   clang++ -std=c++17 -Wall -Wextra -o /tmp/test_gait_fsm tests/test_gait_fsm.cpp && /tmp/test_gait_fsm
#include "../SAKER_Sender/GaitFsm.h"

#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;
#define CHECK(cond, ...)                         \
  do {                                           \
    if (!(cond)) {                               \
      failures++;                                \
      std::printf("FAIL %s:%d  ", __FILE__, __LINE__); \
      std::printf(__VA_ARGS__);                  \
      std::printf("\n");                         \
    }                                            \
  } while (0)

constexpr uint32_t SAMPLE_MS = 10;
constexpr double STRIDE_S = 1.1;  // heel strike at t = 0, 1.1, 2.2, ...
constexpr double PI = 3.14159265358979;

// Small deterministic noise in [-amp, amp].
static float noise(float amp) {
  static uint32_t state = 12345;
  state = state * 1664525u + 1013904223u;
  return amp * ((state >> 8) / float(1 << 24) * 2 - 1);
}

// Shank sagittal rate (deg/s, + = swinging forward) at time t within a stride.
static float shankRate(double t) {
  t = std::fmod(t, STRIDE_S);
  if (t < 0.08) return -40 * (t / 0.08) - 120 * std::sin(PI * t / 0.08);     // heel-strike dip
  if (t < 0.50) return -40;                                                    // mid-stance
  if (t < 0.66) return -40 - 140 * std::sin(PI * (t - 0.50) / 0.16);           // push-off dip
  const double s = (t - 0.66) / 0.44;                                          // swing
  return 350 * std::sin(PI * s) - 40 * (1 - s);
}

static bool inSwing(double t) { return std::fmod(t, STRIDE_S) >= 0.62; }

constexpr size_t MAX_N = 2000;
static float shank[MAX_N][3], thigh[MAX_N][3], scratch[MAX_N];

// Fills n samples with the sagittal signal on `axis` flipped by `sign`.
static void makeWalk(size_t n, int axis, int sign) {
  for (size_t i = 0; i < n; i++) {
    const double t = i * SAMPLE_MS / 1000.0;
    for (int a = 0; a < 3; a++) {
      shank[i][a] = noise(5);
      thigh[i][a] = noise(5);
    }
    shank[i][axis] = sign * shankRate(t) + noise(5);
    // Thigh swings forward with the shank, on a different axis, opposite sign.
    thigh[i][(axis + 1) % 3] = -sign * (0.5f * shankRate(t)) + noise(5);
  }
}

static void testCalibrationFindsAxisAndSign() {
  const size_t n = 1200;  // 12 s
  makeWalk(n, 1, -1);
  const WalkCalResult r = analyseWalk(shank, thigh, n, SAMPLE_MS, scratch);

  CHECK(r.ok, "calibration failed: %s", r.message);
  CHECK(r.shank.axis == 1 && r.shank.sign == -1, "shank axis %d sign %d", r.shank.axis, r.shank.sign);
  CHECK(r.thigh.axis == 2 && r.thigh.sign == 1, "thigh axis %d sign %d", r.thigh.axis, r.thigh.sign);
  CHECK(r.steps >= 10 && r.steps <= 11, "steps %d", r.steps);
  CHECK(r.swingPeak > 250 && r.swingPeak < 360, "swing peak %.0f", r.swingPeak);
  CHECK(r.thresholds.pushOffEnter < -40, "push-off threshold %.0f must sit below mid-stance (-40)",
        r.thresholds.pushOffEnter);
  std::printf("calibration: shank axis %d sign %+d, thigh axis %d sign %+d, %d steps, "
              "swing %.0f, push-off %.0f -> thresholds swing %.0f push-off %.0f heel %.0f\n",
              r.shank.axis, r.shank.sign, r.thigh.axis, r.thigh.sign, r.steps, r.swingPeak,
              r.pushOffPeak, r.thresholds.swingEnter, r.thresholds.pushOffEnter,
              r.thresholds.heelStrike);
}

static void testStandingStillGivesNoSteps() {
  const size_t n = 1200;
  for (size_t i = 0; i < n; i++)
    for (int a = 0; a < 3; a++) { shank[i][a] = noise(3); thigh[i][a] = noise(3); }
  const WalkCalResult r = analyseWalk(shank, thigh, n, SAMPLE_MS, scratch);
  CHECK(!r.ok, "standing still should fail calibration");
}

static void testFsmFollowsEveryStride() {
  // Calibrate on 12 s, then run 20 strides through the FSM.
  const size_t calN = 1200;
  makeWalk(calN, 0, 1);
  const WalkCalResult cal = analyseWalk(shank, thigh, calN, SAMPLE_MS, scratch);
  CHECK(cal.ok, "calibration failed: %s", cal.message);

  GaitFsm fsm;
  fsm.thresholds = cal.thresholds;
  fsm.reset(0);

  const int strides = 20;
  const size_t n = size_t(strides * STRIDE_S * 1000 / SAMPLE_MS);
  std::vector<GaitEvent> events;
  std::vector<double> heelStrikeTimes;
  for (size_t i = 0; i < n; i++) {
    const double t = i * SAMPLE_MS / 1000.0;
    const float thighMag = inSwing(t) ? 120 : 30;
    const GaitEvent e = fsm.update(shankRate(t) + noise(5), thighMag, uint32_t(i * SAMPLE_MS));
    if (e == EVENT_NONE) continue;
    events.push_back(e);
    if (e == EVENT_HEEL_STRIKE) heelStrikeTimes.push_back(t);
  }

  int timeouts = 0;
  for (GaitEvent e : events) timeouts += (e == EVENT_TIMEOUT);
  CHECK(timeouts == 0, "%d timeouts", timeouts);

  // Every stride must produce PUSH_OFF, TOE_OFF, MID_SWING, HEEL_STRIKE in that order.
  const GaitEvent expected[4] = {EVENT_PUSH_OFF, EVENT_TOE_OFF, EVENT_MID_SWING, EVENT_HEEL_STRIKE};
  CHECK(events.size() == size_t(strides) * 4 || events.size() == size_t(strides) * 4 - 1,
        "%zu events for %d strides", events.size(), strides);
  for (size_t i = 0; i < events.size(); i++) {
    CHECK(events[i] == expected[i % 4], "event %zu is %s, expected %s", i,
          gaitEventName(events[i]), gaitEventName(expected[i % 4]));
  }

  // Heel strike detected within 40 ms after the true one (t = k * STRIDE_S).
  double worst = 0;
  for (double t : heelStrikeTimes) {
    const double late = t - std::round(t / STRIDE_S) * STRIDE_S;
    worst = std::max(worst, std::fabs(late));
    CHECK(late >= -0.005 && late <= 0.040, "heel strike at %.3f s is %.0f ms off", t, late * 1000);
  }
  std::printf("fsm: %zu events over %d strides, %zu heel strikes, worst timing %.0f ms\n",
              events.size(), strides, heelStrikeTimes.size(), worst * 1000);
}

static void testStandingStillStaysInStance() {
  GaitFsm fsm;
  fsm.reset(0);
  int events = 0;
  for (uint32_t ms = 0; ms < 10000; ms += SAMPLE_MS) events += fsm.update(noise(8), 5, ms) != EVENT_NONE;
  CHECK(events == 0, "%d events while standing", events);
  CHECK(fsm.phase() == PHASE_STANCE, "phase %s", gaitPhaseName(fsm.phase()));
}

int main() {
  testCalibrationFindsAxisAndSign();
  testStandingStillGivesNoSteps();
  testFsmFollowsEveryStride();
  testStandingStillStaysInStance();
  std::printf(failures ? "\n%d FAILED\n" : "\nall gait tests passed\n", failures);
  return failures ? 1 : 0;
}
