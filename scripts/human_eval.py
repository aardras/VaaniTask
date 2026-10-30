"""Blind native-speaker evaluation of reply mode.

1) Export a blind rating sheet (paired: each sampled utterance appears once per system):
       python scripts/human_eval.py export
   -> results/human_eval/ratings.csv   (open in Excel; fill 3 columns; Save As "CSV UTF-8")
      results/human_eval/key.csv       (which system produced each row; don't open while rating)
      results/human_eval/GUIDE.md      (rating definitions)
2) Analyse after rating:
       python scripts/human_eval.py analyze
   -> per-system / per-condition human scores with bootstrap CIs, paired baseline-vs-improved
      differences, and agreement (Cohen's kappa) between the human and the LLM judge.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.response_metrics import cohen_kappa  # noqa: E402
from src.stats import mean_ci, paired_diff_ci  # noqa: E402

OUT = Path("results/human_eval")
SYSTEMS = {"baseline_reply": "baseline", "improved_reply": "improved"}
LANG = {"hi_in": "Hindi", "ml_in": "Malayalam"}
CORRECT = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
RELEVANT = {"on_topic": 1.0, "partial": 0.5, "off_topic": 0.0}

GUIDE = """# Rating guide: reply mode

Each row is one reply from a voice assistant. Judge the reply against **what the person actually said**
(column `what_was_said`; `english_meaning` is there to help). The assistant only heard the audio, so it may
have misheard; that counts against it. You don't know which system wrote which reply. Rate each row on its own.

## correctness (response quality): `correct` / `partial` / `incorrect`
Is this a good, correct response to what the person actually said?
- **correct**: a useful reply that responds correctly to what was said.
- **partial**: a reasonable attempt, but part of it is wrong, misstated or unhelpful.
- **incorrect**: wrong, misstates what the person said, nonsense, or a non-answer
  ("I didn't understand, please repeat" counts as **incorrect**: the person got no response).

## relevance: `on_topic` / `partial` / `off_topic`
Does the reply respond to what the person actually said?
- **on_topic**: engages with the actual subject and content.
- **partial**: right general topic, but responds to a detail they didn't say.
- **off_topic**: about something else, or a generic "please repeat" that doesn't engage with the content.

## understandability: `1` to `5`
As a native speaker, how easy is **the reply text itself** to read and understand, and how natural is it?
This is about the assistant's language, not about how well it understood the person: that is
what correctness and relevance measure. A fluent reply about the wrong topic is still a 5 here.
- **5**: fluent and natural; reads like a native speaker wrote it.
- **4**: clear, with minor awkwardness (odd word choice, slightly unnatural phrasing).
- **3**: understandable with some effort; clearly non-native or awkward.
- **2**: hard to understand; meaning only partly recoverable.
- **1**: incomprehensible, garbled, or not in the right language.
Rate the language itself, not whether it's correct or relevant.

## Saving
Fill only the three columns, then **File > Save As > "CSV UTF-8 (Comma delimited)"** (plain CSV can garble Malayalam).
"""


def export(n_per_cell: int, seed: int):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "ratings.csv").exists():
        sys.exit(f"{OUT / 'ratings.csv'} exists; refusing to overwrite ratings")
    runs = {k: pd.read_json(f"results/{k}/scored.jsonl", lines=True, dtype={"snr": str}).set_index("item_id")
            for k in SYSTEMS}
    common = runs["baseline_reply"].index.intersection(runs["improved_reply"].index)
    base = runs["baseline_reply"].loc[common]
    picked = base.groupby(["lang", "snr"], group_keys=False).apply(
        lambda g: g.sample(min(n_per_cell, len(g)), random_state=seed)).index
    rows = []
    for item in picked:
        for run, name in SYSTEMS.items():
            r = runs[run].loc[item]
            rows.append({"item_id": item, "system": name, "language": LANG[r.lang], "snr": r.snr,
                         "what_was_said": r.reference, "english_meaning": r.english_gold, "reply": r.response})
    d = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    d.insert(0, "row", np.arange(1, len(d) + 1))
    d[["row", "item_id", "system"]].to_csv(OUT / "key.csv", index=False)
    sheet = d[["row", "language", "what_was_said", "english_meaning", "reply"]].assign(
        correctness="", relevance="", understandability="")
    sheet.to_csv(OUT / "ratings.csv", index=False, encoding="utf-8-sig")  # BOM: Excel shows Indic text
    (OUT / "GUIDE.md").write_text(GUIDE, encoding="utf-8")
    print(f"wrote {OUT / 'ratings.csv'} ({len(sheet)} rows = {len(picked)} utterances x 2 systems), "
          f"{OUT / 'key.csv'}, {OUT / 'GUIDE.md'}")


def analyze():
    h = pd.read_csv(OUT / "ratings.csv", encoding="utf-8-sig").merge(pd.read_csv(OUT / "key.csv"), on="row")
    for c in ["correctness", "relevance"]:
        h[c] = h[c].astype(str).str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")
    h["understandability"] = pd.to_numeric(h.understandability, errors="coerce")
    bad = h[~h.correctness.isin(list(CORRECT)) | ~h.relevance.isin(list(RELEVANT)) | ~h.understandability.between(1, 5)]
    if len(bad):
        print(f"WARNING: {len(bad)} rows incomplete or invalid, excluded: rows {bad.row.tolist()}")
    h = h.drop(bad.index)
    h["correct_score"] = h.correctness.map(CORRECT)
    h["relevance_score"] = h.relevance.map(RELEVANT)
    h["correct"] = h.correctness == "correct"
    h["on_topic"] = h.relevance == "on_topic"
    meta = pd.concat([pd.read_json(f"results/{k}/scored.jsonl", lines=True, dtype={"snr": str})
                      .assign(system=v) for k, v in SYSTEMS.items()])
    h = h.merge(meta[["item_id", "system", "lang", "snr", "rj_relevance", "rj_misattribution",
                      "rj_factual_halluc", "rj_clarification", "acceptable"]], on=["item_id", "system"])

    lines = ["# Native-speaker evaluation of reply mode (blind; auto-generated by scripts/human_eval.py)\n",
             f"Rated replies: {len(h)} ({h.item_id.nunique()} utterances x 2 systems). Single native-speaker "
             "annotator for both languages. Response quality = % rated `correct` (a good, correct response; "
             "non-answers such as \"please repeat\" rated incorrect); relevance = % on-topic; "
             "understandability = mean 1-5. Brackets: bootstrap 95% CI.\n",
             "| System | Language | Condition | n | Good response % | On-topic % | Understandability (1-5) |",
             "|---|---|---|---|---|---|---|"]
    keys = [("ALL", "ALL")] + sorted({(l, s) for l, s in zip(h.lang, h.snr)})
    for sysname in SYSTEMS.values():
        for lang, snr in keys:
            g = h[(h.system == sysname) & ((h.lang == lang) | (lang == "ALL")) & ((h.snr == snr) | (snr == "ALL"))]
            if g.empty:
                continue
            c, u = mean_ci(g.correct), mean_ci(g.understandability)
            o = mean_ci(g.on_topic)
            cond = "all" if snr == "ALL" else ("clean" if snr == "clean" else f"{snr} dB")
            lines.append(f"| {sysname} | {LANG.get(lang, 'both')} | {cond} | {len(g)} | {100 * c[0]:.0f} "
                         f"[{100 * c[1]:.0f}, {100 * c[2]:.0f}] | {100 * o[0]:.0f} [{100 * o[1]:.0f}, {100 * o[2]:.0f}] "
                         f"| {u[0]:.2f} [{u[1]:.2f}, {u[2]:.2f}] |")
    p = h.pivot_table(index="item_id", columns="system",
                      values=["correct", "on_topic", "understandability"], aggfunc="first").dropna()
    lines += ["", "**Improved − baseline, paired on the same utterances**\n",
              "| Measure | Difference | 95% CI | p |", "|---|---|---|---|"]
    for col, label, scale in [("correct", "Good response (pts)", 100), ("on_topic", "On-topic (pts)", 100),
                              ("understandability", "Understandability (1-5)", 1)]:
        d = paired_diff_ci(p[(col, "baseline")].astype(float), p[(col, "improved")].astype(float))
        lines.append(f"| {label} | {scale * d['diff']:+.2f} | [{scale * d['lo']:+.2f}, {scale * d['hi']:+.2f}] "
                     f"| {d['p']:.3f} |")
    # The human's correctness is overall response quality, so it is compared with the judge's
    # overall verdict (acceptable), not with its hallucination flags (a different construct).
    lines += ["", "**Agreement between the human and the LLM judge** (Cohen's kappa)\n",
              "| Criterion | All | Hindi | Malayalam |", "|---|---|---|---|"]
    for label, a, b in [("Relevance (3-level)", h.relevance, h.rj_relevance.astype(str)),
                        ("Response quality: human 'correct' vs judge 'acceptable'", h.correct, h.acceptable.astype(bool))]:
        cells = [f"{cohen_kappa(a, b):.2f}"] + [f"{cohen_kappa(a[h.lang == l], b[h.lang == l]):.2f}" for l in LANG]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    clar = h[h.rj_clarification.astype(bool)]
    lines += ["", f"Replies the judge flagged as asking the user to repeat: {len(clar)}; human ratings for them: "
              + ", ".join(f"{k} {v}" for k, v in clar.correctness.value_counts().items())
              + ". The human did not rate hallucination separately, so the judge's misattribution and "
              "factual-hallucination labels are not validated by this evaluation."]
    (OUT / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["export", "analyze"])
    ap.add_argument("--per-cell", type=int, default=6, help="utterances per language x SNR (x2 systems)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    export(args.per_cell, args.seed) if args.mode == "export" else analyze()


if __name__ == "__main__":
    main()
