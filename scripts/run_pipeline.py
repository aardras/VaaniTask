"""Run speech -> ASR -> LLM over the manifest.

    python scripts/run_pipeline.py --config configs/baseline.yaml [--limit 5]
    python scripts/run_pipeline.py --config configs/baseline.yaml --oracle   # gold transcript -> LLM
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipeline import reply_subset, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--limit", type=int, help="items per (lang, snr) group, for smoke tests")
    ap.add_argument("--asr-only", action="store_true", help="skip the LLM stage (ASR ablation rows)")
    ap.add_argument("--task", choices=["restate", "reply"], default="restate")
    ap.add_argument("--reply-per-lang", type=int, default=40,
                    help="reply mode: utterances per language (each at clean and 0 dB)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    manifest = pd.read_csv(Path(cfg["data"]["root"]) / "manifest.csv", dtype={"snr": str})
    manifest = manifest[manifest.lang.isin(cfg["data"]["languages"])]  # e.g. ml-only control run
    asr_from = None
    if args.task == "reply":
        # Same ASR transcripts as the restatement run; only the LLM task changes.
        manifest = reply_subset(manifest, args.reply_per_lang, cfg["seed"])
        asr_from = Path("results") / cfg["run_name"]
    name = (cfg["run_name"] + ("_oracle" if args.oracle else "") + ("_reply" if args.task == "reply" else "")
            + (f"_smoke{args.limit}" if args.limit else ""))
    df = run(cfg, manifest, Path("results") / name, oracle=args.oracle, limit=args.limit,
             asr_only=args.asr_only, task=args.task, asr_from=None if args.oracle else asr_from)
    print(f"results/{name}: {len(df)} items")


if __name__ == "__main__":
    main()
