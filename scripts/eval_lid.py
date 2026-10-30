"""Evaluate automatic language identification on every manifest item.

    python scripts/eval_lid.py     ->  results/lid/{lid.jsonl, metrics.csv}
Accuracy (with bootstrap 95% CI), confusion and latency per language x SNR.
"""
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import load_audio  # noqa: E402
from src.lid import LanguageID  # noqa: E402
from src.stats import mean_ci  # noqa: E402


def main():
    out = Path("results/lid")
    out.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv("data/manifest.csv", dtype={"snr": str})
    lid = LanguageID()
    lid.detect(load_audio(m.audio_path.iloc[0]))  # warm-up
    rows = []
    for r in tqdm(m.itertuples(), total=len(m), desc="LID"):
        pred, dist, secs = lid.detect(load_audio(r.audio_path))
        rows.append({"item_id": r.item_id, "lang": r.lang, "snr": r.snr, "pred": pred,
                     "p_true": dist[r.lang], "lid_s": secs, "duration_s": r.duration_s})
    d = pd.DataFrame(rows)
    d.to_json(out / "lid.jsonl", orient="records", lines=True)
    d["correct"] = d.pred == d.lang
    agg = []
    for (lang, snr), g in [(("ALL", "ALL"), d)] + list(d.groupby(["lang", "snr"], sort=False)):
        acc, lo, hi = mean_ci(g.correct)
        agg.append({"lang": lang, "snr": snr, "n": len(g), "accuracy": acc, "acc_lo": lo, "acc_hi": hi,
                    "mean_p_true": g.p_true.mean(), "lid_s_p50": g.lid_s.median(),
                    "lid_rtf_p50": (g.lid_s / g.duration_s).median()})
    agg = pd.DataFrame(agg)
    agg.to_csv(out / "metrics.csv", index=False)
    print(agg.round(3).to_string(index=False))
    print("\nconfusion (rows = true, cols = predicted):")
    print(pd.crosstab(d.lang, d.pred).to_string())
    wrong = d[~d.correct]
    if len(wrong):
        print("\nmisidentified:", json.dumps(wrong[["item_id", "pred", "p_true"]].to_dict("records")[:10]))


if __name__ == "__main__":
    main()
