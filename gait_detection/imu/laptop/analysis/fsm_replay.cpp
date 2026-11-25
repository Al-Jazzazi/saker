// Replays samples through the board's own GaitFsm (SAKER_Sender/GaitFsm.h).
// Args: swingEnter pushOffEnter heelStrike. stdin: "t_ms shankRate thighMag" per line.
// stdout: "phase event" per line (GaitPhase / GaitEvent enum values).
// Build: clang++ -std=c++17 -O2 -I ../../SAKER_Sender fsm_replay.cpp -o fsm_replay
#include <cstdio>
#include <cstdlib>
#include "GaitFsm.h"
int main(int argc, char **argv) {
  if (argc != 4) { fprintf(stderr, "usage: fsm_replay swingEnter pushOffEnter heelStrike\n"); return 2; }
  GaitFsm fsm;
  fsm.thresholds.swingEnter = atof(argv[1]);
  fsm.thresholds.pushOffEnter = atof(argv[2]);
  fsm.thresholds.heelStrike = atof(argv[3]);
  unsigned t;
  float s, m;
  bool first = true;
  while (scanf("%u %f %f", &t, &s, &m) == 3) {
    if (first) { fsm.reset(t); first = false; }
    GaitEvent e = fsm.update(s, m, t);
    printf("%d %d\n", (int)fsm.phase(), (int)e);
  }
}
