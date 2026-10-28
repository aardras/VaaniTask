#!/usr/bin/env bash
# Run the remaining experiment queue strictly one job at a time (timed runs must run alone).
# Stops at the first failing step; every stage resumes from its JSONL, so just rerun.
#   bash scripts/run_queue.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_DISABLE_SYMLINKS_WARNING=1 PYTHONIOENCODING=utf-8
PY=.venv/Scripts/python.exe
mkdir -p logs

step() {  # step <name> <cmd...>
  local name=$1; shift
  echo "$(date +%H:%M:%S) START $name"
  if "$@" > "logs/$name.log" 2>&1; then
    echo "$(date +%H:%M:%S) DONE  $name"
  else
    echo "$(date +%H:%M:%S) FAIL  $name (see logs/$name.log)"; tail -5 "logs/$name.log"; exit 1
  fi
}

# Wait for the baseline LLM stage that is already running in another process.
while [ "$(wc -l < results/baseline/llm.jsonl 2>/dev/null || echo 0)" -lt 720 ] || \
      powershell.exe -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_pipeline.py*' }) { exit 0 } else { exit 1 }"; do
  sleep 30
done
echo "$(date +%H:%M:%S) baseline LLM stage finished"

step baseline_merge     $PY scripts/run_pipeline.py --config configs/baseline.yaml   # writes predictions.jsonl
step eval_baseline      $PY scripts/evaluate.py --config configs/baseline.yaml --run results/baseline
step oracle             $PY scripts/run_pipeline.py --config configs/baseline.yaml --oracle
step eval_oracle        $PY scripts/evaluate.py --config configs/baseline.yaml --run results/baseline_oracle
step adapted_hf         $PY scripts/run_pipeline.py --config configs/adapted_hf.yaml --asr-only
step eval_adapted_hf    $PY scripts/evaluate.py --config configs/adapted_hf.yaml --run results/adapted_hf
step improved           $PY scripts/run_pipeline.py --config configs/improved.yaml
step eval_improved      $PY scripts/evaluate.py --config configs/improved.yaml --run results/improved
step ml_medium_control  $PY scripts/run_pipeline.py --config configs/ml_medium_control.yaml --asr-only
step eval_ml_control    $PY scripts/evaluate.py --config configs/ml_medium_control.yaml --run results/ml_medium_control
step report             $PY scripts/make_report.py
echo "$(date +%H:%M:%S) ALL DONE"
