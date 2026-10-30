"""Human validation of the LLM judge (restatement or reply runs).

1) Export 40 scored items (judge columns hidden), stratified by language x SNR:
       python scripts/label_sample.py export --run results/improved_reply
   Fill the empty `human_*` columns in <run>/human_labels.csv using the definitions in
   src/metrics/response_metrics.py:
     restatement runs: human_label (correct/partial/incorrect), human_halluc (true/false)
     reply runs:       human_relevance (on_topic/partial/off_topic), human_misattribution,
                       human_factual_halluc, human_clarification (true/false)
2) Agreement with the judge, overall and per language (Cohen's kappa):
       python scripts/label_sample.py agree --run results/improved_reply
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.response_metrics import cohen_kappa  # noqa: E402

# human column -> (judge column, is_boolean)
FIELDS = {
    "restate": {"human_label": ("judge_label", False), "human_halluc": ("judge_halluc", True)},
    "reply": {"human_relevance": ("rj_relevance", False),
              "human_misattribution": ("rj_misattribution", True),
              "human_factual_halluc": ("rj_factual_halluc", True),
              "human_clarification": ("rj_clarification", True)},
}


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "agree"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()
    run = Path(args.run)
    path = run / "human_labels.csv"
    scored = pd.read_json(run / "scored.jsonl", lines=True, dtype={"snr": str})
    kind = "reply" if "rj_relevance" in scored else "restate"
    fields = FIELDS[kind]
    if args.mode == "export":
        if path.exists():
            sys.exit(f"{path} exists; refusing to overwrite hand labels")
        groups = scored.groupby(["lang", "snr"])
        per = max(1, args.n // groups.ngroups)
        s = groups.sample(min(per, groups.size().min()), random_state=0)
        s = s[["item_id", "lang", "snr", "reference", "english_gold", "response"]]
        s.assign(**{c: "" for c in fields}).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"wrote {path} ({len(s)} items, {kind} rubric: fill {', '.join(fields)})")
        return
    h = pd.read_csv(path, dtype={"snr": str}).merge(
        scored[["item_id"] + [j for j, _ in fields.values()]], on="item_id")
    h = h[h[list(fields)[0]].notna()]
    print(f"{kind} run, n={len(h)} hand-labelled")
    for hum, (jud, is_bool) in fields.items():
        a = _bool(h[hum]) if is_bool else h[hum].astype(str).str.strip().str.lower()
        b = h[jud].astype(bool) if is_bool else h[jud].astype(str)
        line = f"{hum:22s} agreement={(a == b).mean():.3f} kappa={cohen_kappa(a, b):.3f}"
        for lang, idx in h.groupby("lang").groups.items():
            line += f" | {lang} kappa={cohen_kappa(a[idx], b[idx]):.3f}"
        print(line)


if __name__ == "__main__":
    main()
