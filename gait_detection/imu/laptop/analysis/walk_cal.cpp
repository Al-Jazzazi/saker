// Runs the board's analyseWalk() on stdin samples: "sx sy sz tx ty tz" (bias-corrected deg/s).
// stdout: status swingPeak pushOffPeak swingEnter pushOffEnter heelStrike steps shankAxis thighAxis
// Build: clang++ -std=c++17 -O2 -I ../../SAKER_Sender walk_cal.cpp -o walk_cal
#include <array>
#include <cstdio>
#include <vector>
#include "GaitFsm.h"
int main() {
  std::vector<std::array<float, 3>> s, t;
  float a, b, c, d, e, f;
  while (scanf("%f %f %f %f %f %f", &a, &b, &c, &d, &e, &f) == 6) { s.push_back({a, b, c}); t.push_back({d, e, f}); }
  std::vector<float> scratch(s.size());
  WalkCalResult r = analyseWalk(reinterpret_cast<const float (*)[3]>(s.data()), reinterpret_cast<const float (*)[3]>(t.data()),
                                s.size(), 10, scratch.data());
  printf("%s %.1f %.1f %.1f %.1f %.1f %d %c%c %c%c\n", r.ok ? "ok" : "fail", r.swingPeak, r.pushOffPeak,
         r.thresholds.swingEnter, r.thresholds.pushOffEnter, r.thresholds.heelStrike, r.steps,
         "XYZ"[r.shank.axis], r.shank.sign > 0 ? '+' : '-', "XYZ"[r.thigh.axis], r.thigh.sign > 0 ? '+' : '-');
}
