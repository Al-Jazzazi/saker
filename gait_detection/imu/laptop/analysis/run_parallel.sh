#!/bin/zsh
# benchmark, size study and interpretability side by side; then analysis and report
cd "$(dirname "$0")"
export VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1
PY=${1:-python3}
WORKERS=5 $PY benchmark.py > ../results/benchmark.log 2>&1 &
WORKERS=2 $PY size_study.py > ../results/size_study.log 2>&1 &
WORKERS=2 $PY interpret.py > ../results/interpret.log 2>&1 &
wait
$PY analyze.py > ../results/analyze.log 2>&1 && $PY interpret.py errors >> ../results/analyze.log 2>&1 && $PY export_report.py >> ../results/analyze.log 2>&1 && $PY write_report.py >> ../results/analyze.log 2>&1 && echo ALL_DONE >> ../results/analyze.log
