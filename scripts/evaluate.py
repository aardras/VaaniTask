"""Score a run: per-item metrics -> scored.jsonl, aggregates with 95% CIs -> metrics.csv.

    python scripts/evaluate.py --config configs/baseline.yaml --run results/baseline
"""
import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm import LLM  # noqa: E402
from src.metrics.asr_metrics import utterance_errors  # noqa: E402
from src.metrics.latency import summarize  # noqa: E402
from src.metrics.response_metrics import (LABEL_SCORE, LaBSE, chrf,  # noqa: E402
                                          heuristic_hallucination, judge_one, off_script)
from src.stats import mean_ci, ratio_ci  # noqa: E402


def score_items(cfg: dict, run: Path) -> pd.DataFrame:
    df = pd.read_json(run / "predictions.jsonl", lines=True, dtype={"snr": str})
    # ASR errors (hypothesis vs reference) ...
    err = pd.DataFrame([utterance_errors(r.reference, r.hypothesis, r.lang) for r in df.itertuples()])
    if "response" not in df.columns:  # --asr-only run: ASR metrics + latency only
        return pd.concat([df, err], axis=1)
    # ... and the same errors measured on the LLM's restatement, to see whether it repairs ASR.
    resp = pd.DataFrame([utterance_errors(r.reference, r.response, r.lang) for r in df.itertuples()])
    resp = resp[["word_errors", "char_errors"]].add_prefix("resp_")
    df = pd.concat([df, err, resp], axis=1)
    df["chrf"] = chrf(df.response.tolist(), df.reference.tolist())
    df["off_script"] = [off_script(r, l) for r, l in zip(df.response, df.lang)]
    labse = LaBSE()
    df["labse"] = labse.similarity(df.response.tolist(), df.english_gold.tolist())  # cross-lingual
    del labse
    heur = pd.DataFrame([heuristic_hallucination(r.response, r.reference, r.english_gold)
                         for r in df.itertuples()])
    df = pd.concat([df, heur], axis=1)

    # Judge results are cached by (item_id, response) so re-scoring never re-runs the 7B model.
    cache_path = run / "judge_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in cache_path.open(encoding="utf-8"):
            d = json.loads(l)
            cache[(d.pop("item_id"), d.pop("response"))] = d
    todo = df[[(i, r) not in cache for i, r in zip(df.item_id, df.response)]]
    if len(todo):
        gc.collect()
        torch.cuda.empty_cache()
        jc = cfg["judge"]
        judge = LLM(jc["model"], jc["load_in_4bit"], quantize_lm_head=jc.get("quantize_lm_head", False))
        with cache_path.open("a", encoding="utf-8") as f:
            for t in tqdm(todo.itertuples(), total=len(todo), desc="judge"):
                i, r = t.item_id, t.response
                res = judge_one(judge, r, t.reference, t.english_gold, t.lang)
                cache[(i, r)] = res
                f.write(json.dumps({"item_id": i, "response": r, **res}, ensure_ascii=False) + "\n")
                f.flush()
    jd = pd.DataFrame([cache[(i, r)] for i, r in zip(df.item_id, df.response)])
    df = pd.concat([df, jd], axis=1)
    df["judge_score"] = df.judge_label.map(LABEL_SCORE)
    df["judge_correct"] = df.judge_label == "correct"
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", "ALL", df)] + [(l, s, g) for (l, s), g in df.groupby(["lang", "snr"], sort=False)] \
        + [(l, "ALL", g) for l, g in df.groupby("lang", sort=False)]
    for lang, snr, g in groups:
        row = {"lang": lang, "snr": snr, "n": len(g)}
        for name, (num, den) in {"wer": ("word_errors", "ref_words"),
                                 "cer": ("char_errors", "ref_chars"),
                                 "resp_wer": ("resp_word_errors", "ref_words"),
                                 "resp_cer": ("resp_char_errors", "ref_chars")}.items():
            if num in g:
                row[name], row[f"{name}_lo"], row[f"{name}_hi"] = ratio_ci(g[num], g[den])
        for col in ["degenerate", "off_script", "heur_runaway", "judge_correct", "judge_score", "judge_halluc",
                    "heur_halluc", "chrf", "labse"]:
            if col in g:
                row[col], row[f"{col}_lo"], row[f"{col}_hi"] = mean_ci(g[col])
        row.update(summarize(g))
        if "judge_parse_ok" in g:
            row["judge_parse_fail"] = int((~g.judge_parse_ok).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    run = Path(args.run)
    df = score_items(cfg, run)
    df.to_json(run / "scored.jsonl", orient="records", lines=True, force_ascii=False)
    agg = aggregate(df)
    agg.to_csv(run / "metrics.csv", index=False)
    cols = ["lang", "snr", "n", "wer", "cer", "resp_cer", "degenerate", "off_script", "judge_correct", "judge_halluc", "heur_halluc",
            "chrf", "labse", "asr_rtf_mean", "e2e_s_mean"]
    print(agg[[c for c in cols if c in agg]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
