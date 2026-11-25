#!/bin/zsh
# Runs everything after the manual annotation, in order. Local only: no AI agents.
#   1. merge the by-eye marks into one file per recording   (annotations/merged/)
#   2. benchmark: FSM vs ML vs ML + online learning          (results/benchmark/raw_results.npz)
#   3. analysis: scores, per task/person, bootstrap CIs      (results/benchmark/summary.json + CSVs)
#   4. report tables + figures                               (results/benchmark/report/, REPORT.md)
# Usage: ./run_all.sh [PYTHON]   (PYTHON needs numpy scipy scikit-learn matplotlib)
set -e
cd "$(dirname "$0")"
PY=${1:-python3}
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1   # one BLAS thread per worker process

# the board's own C++ (GaitFsm.h) compiled for the laptop
[ -x fsm_replay ] || clang++ -std=c++17 -O2 -I ../../SAKER_Sender fsm_replay.cpp -o fsm_replay
[ -x walk_cal ]   || clang++ -std=c++17 -O2 -I ../../SAKER_Sender walk_cal.cpp -o walk_cal

echo "== 1/4 merge marks";   $PY labels.py merge
echo "== 2/4 benchmark";     $PY benchmark.py
echo "== 3/4 analysis";      $PY analyze.py
echo "== 4/6 size study";    $PY size_study.py
echo "== 5/6 interpretability"; $PY interpret.py
echo "== 6/6 report";        $PY export_report.py && $PY write_report.py
echo "done: see ../results/benchmark/REPORT.md"
