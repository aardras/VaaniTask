#!/usr/bin/env bash
# Reply-mode experiments: LLM replies on the fixed 160-item subset (40 utterances per language
# x clean/0 dB), reusing each run's existing ASR transcripts; then rubric scoring.
# One job at a time; stops at the first failure; every stage resumes, so just rerun.
#   bash scripts/run_reply_queue.sh
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

step reply_baseline      $PY scripts/run_pipeline.py --config configs/baseline.yaml --task reply
step reply_improved      $PY scripts/run_pipeline.py --config configs/improved.yaml --task reply
step reply_oracle        $PY scripts/run_pipeline.py --config configs/baseline.yaml --task reply --oracle
step eval_reply_baseline $PY scripts/evaluate_reply.py --config configs/baseline.yaml --run results/baseline_reply
step eval_reply_improved $PY scripts/evaluate_reply.py --config configs/improved.yaml --run results/improved_reply
step eval_reply_oracle   $PY scripts/evaluate_reply.py --config configs/baseline.yaml --run results/baseline_oracle_reply
echo "$(date +%H:%M:%S) ALL DONE"
