"""Score a reply-mode run with the grounded rubric -> scored.jsonl, metrics.csv.

    python scripts/evaluate_reply.py --config configs/improved.yaml --run results/improved_reply

Criteria are defined in src/metrics/response_metrics.py (reply-mode section); every
one is judged against what the user ACTUALLY said, never against the ASR transcript.
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
from src.metrics.response_metrics import (REPLY_JUDGE_VERSION, RELEVANCE_SCORE, LaBSE,  # noqa: E402
                                          judge_reply_one, off_script)
from src.stats import mean_ci, ratio_ci  # noqa: E402

BOOL_COLS = ["acceptable", "on_topic", "rj_misattribution", "rj_factual_halluc",
             "rj_clarification", "off_script"]


def score_items(cfg: dict, run: Path) -> pd.DataFrame:
    df = pd.read_json(run / "predictions.jsonl", lines=True, dtype={"snr": str})
    err = pd.DataFrame([utterance_errors(r.reference, r.hypothesis, r.lang) for r in df.itertuples()])
    df = pd.concat([df, err], axis=1)
    df["off_script"] = [off_script(r, l) for r, l in zip(df.response, df.lang)]
    labse = LaBSE()
    # Judge-free relevance signal: a reply should be semantically related to what was said.
    df["labse_vs_said"] = labse.similarity(df.response.tolist(), df.english_gold.tolist())
    del labse

    cache_path = run / f"judge_cache_{REPLY_JUDGE_VERSION}.jsonl"
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
            for t in tqdm(todo.itertuples(), total=len(todo), desc="reply judge"):
                res = judge_reply_one(judge, t.response, t.reference, t.english_gold, t.lang)
                cache[(t.item_id, t.response)] = res
                f.write(json.dumps({"item_id": t.item_id, "response": t.response, **res},
                                   ensure_ascii=False) + "\n")
                f.flush()
    jd = pd.DataFrame([cache[(i, r)] for i, r in zip(df.item_id, df.response)])
    df = pd.concat([df, jd], axis=1)
    df["relevance_score"] = df.rj_relevance.map(RELEVANCE_SCORE)
    df["on_topic"] = df.rj_relevance == "on_topic"
    df["acceptable"] = df.on_topic & ~df.rj_misattribution & ~df.rj_factual_halluc & ~df.off_script
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = [("ALL", "ALL", df)] + [(l, s, g) for (l, s), g in df.groupby(["lang", "snr"], sort=False)] \
        + [(l, "ALL", g) for l, g in df.groupby("lang", sort=False)]
    for lang, snr, g in groups:
        row = {"lang": lang, "snr": snr, "n": len(g)}
        row["cer"], row["cer_lo"], row["cer_hi"] = ratio_ci(g.char_errors, g.ref_chars)
        for col in BOOL_COLS + ["relevance_score", "labse_vs_said"]:
            row[col], row[f"{col}_lo"], row[f"{col}_hi"] = mean_ci(g[col])
        row.update(summarize(g))
        row["judge_parse_fail"] = int((~g.rj_parse_ok).sum())
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
    cols = ["lang", "snr", "n", "cer", "acceptable", "on_topic", "rj_misattribution",
            "rj_factual_halluc", "rj_clarification", "off_script", "labse_vs_said", "llm_s_p50"]
    print(agg[[c for c in cols if c in agg]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
