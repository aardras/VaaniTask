"""Human validation of the LLM judge.

1) Export 40 random scored items (judge columns hidden) for hand-labelling:
       python scripts/label_sample.py export --run results/baseline
   Fill `human_label` (correct/partial/incorrect) and `human_halluc` (true/false)
   in results/baseline/human_labels.csv, using the definitions in
   src/metrics/response_metrics.py.
2) Agreement with the judge (Cohen's kappa):
       python scripts/label_sample.py agree --run results/baseline
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.response_metrics import cohen_kappa  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "agree"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    run = Path(args.run)
    path = run / "human_labels.csv"
    scored = pd.read_json(run / "scored.jsonl", lines=True)
    if args.mode == "export":
        if path.exists():
            sys.exit(f"{path} exists; refusing to overwrite hand labels")
        s = scored.sample(args.n, random_state=0)[["item_id", "lang", "snr", "reference", "english_gold", "response"]]
        s.assign(human_label="", human_halluc="").to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {path}")
        return
    h = pd.read_csv(path).merge(scored[["item_id", "judge_label", "judge_halluc"]], on="item_id")
    h = h[h.human_label.notna()]
    h["human_halluc"] = h.human_halluc.astype(str).str.lower().eq("true")
    print(f"n={len(h)}")
    print(f"label  agreement={(h.human_label == h.judge_label).mean():.3f} "
          f"kappa={cohen_kappa(h.human_label, h.judge_label):.3f}")
    print(f"halluc agreement={(h.human_halluc == h.judge_halluc).mean():.3f} "
          f"kappa={cohen_kappa(h.human_halluc, h.judge_halluc):.3f}")


if __name__ == "__main__":
    main()
