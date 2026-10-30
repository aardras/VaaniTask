"""Two-stage runner: ASR over every item, unload, then LLM over every transcript.

The 6 GB GPU must never hold both models at once, so end-to-end latency is the
sum of the per-item stage times (model load/unload excluded). Both stages
append to JSONL and skip finished items, so an interrupted run resumes.
"""
import gc
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from src.asr import load_asr
from src.data import load_audio
from src.llm import LLM


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(l)["item_id"] for l in path.open(encoding="utf-8") if l.strip()}


def _free():
    gc.collect()
    torch.cuda.empty_cache()


def _warmup_flags(df: pd.DataFrame, n: int) -> pd.Series:
    """First n items of each (lang, snr) group are timing warm-ups."""
    return df.groupby(["lang", "snr"]).cumcount() < n


def run_asr(cfg: dict, manifest: pd.DataFrame, out: Path):
    ac = cfg["asr"]
    done = _done_ids(out)
    for lang, df in manifest.groupby("lang", sort=False):
        todo = df[~df.item_id.isin(done)]
        if todo.empty:
            continue
        asr = load_asr(ac["backend"], ac["models"][lang], ac["device"], ac["dtype"])
        asr.transcribe(load_audio(todo.audio_path.iloc[0]), lang)  # discard: CUDA/kernel init
        with out.open("a", encoding="utf-8") as f:
            for r in tqdm(todo.itertuples(), total=len(todo), desc=f"ASR {lang}"):
                text, secs = asr.transcribe(load_audio(r.audio_path), lang)
                f.write(json.dumps({"item_id": r.item_id, "hypothesis": text, "asr_s": secs},
                                   ensure_ascii=False) + "\n")
                f.flush()
        del asr
        _free()


def run_llm(cfg: dict, items: pd.DataFrame, out: Path, text_col: str, task: str = "restate"):
    """items must have item_id, lang and `text_col` (ASR hypothesis or gold transcript)."""
    done = _done_ids(out)
    todo = items[~items.item_id.isin(done)]
    if todo.empty:
        return
    lc = cfg["llm"]
    llm = LLM(lc["model"], lc["load_in_4bit"], lc["max_new_tokens"])
    first = todo.iloc[0]
    llm.respond(first[text_col], first.lang, first.duration_s, task)  # discard: CUDA/kernel init
    with out.open("a", encoding="utf-8") as f:
        for r in tqdm(todo.itertuples(), total=len(todo), desc="LLM"):
            res = llm.respond(getattr(r, text_col), r.lang, r.duration_s, task)
            f.write(json.dumps({"item_id": r.item_id, "response": res["text"],
                                "ttft_s": res["ttft_s"], "llm_s": res["llm_s"],
                                "out_tokens": res["out_tokens"]}, ensure_ascii=False) + "\n")
            f.flush()
    del llm
    _free()


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame([json.loads(l) for l in path.open(encoding="utf-8") if l.strip()])


def reply_subset(manifest: pd.DataFrame, n_per_lang: int, seed: int) -> pd.DataFrame:
    """Fixed subset for reply mode: n utterances per language, each at clean and 0 dB."""
    utts = (manifest[["utt_id", "lang"]].drop_duplicates()
            .groupby("lang", group_keys=False).apply(lambda g: g.sample(n_per_lang, random_state=seed)))
    return manifest[manifest.utt_id.isin(utts.utt_id) & manifest.snr.isin(["clean", "0"])]


def run(cfg: dict, manifest: pd.DataFrame, run_dir: Path, oracle: bool = False,
        limit: int | None = None, asr_only: bool = False, task: str = "restate",
        asr_from: Path | None = None) -> pd.DataFrame:
    """asr_from: reuse ASR outputs (and their timings) from a finished run instead of re-running ASR."""
    run_dir.mkdir(parents=True, exist_ok=True)
    if limit:
        manifest = manifest.groupby(["lang", "snr"], sort=False).head(limit)
    if oracle:
        # Gold transcript straight into the LLM: one item per utterance (noise is irrelevant).
        items = manifest[manifest.snr == "clean"].assign(hypothesis=lambda d: d.reference,
                                                         asr_s=0.0)
        run_llm(cfg, items, run_dir / "llm.jsonl", "reference", task)
        merged = items.merge(read_jsonl(run_dir / "llm.jsonl"), on="item_id")
    else:
        if asr_from is not None:
            src = read_jsonl(asr_from / "asr.jsonl")
            asr = manifest.merge(src, on="item_id")
            assert len(asr) == len(manifest), f"{asr_from} lacks ASR for some items"
        else:
            run_asr(cfg, manifest, run_dir / "asr.jsonl")
            asr = manifest.merge(read_jsonl(run_dir / "asr.jsonl"), on="item_id")
        if asr_only:
            # ASR ablation rows (accuracy / size attribution) skip the slow LLM stage.
            merged = asr.assign(llm_s=0.0, ttft_s=float("nan"), out_tokens=float("nan"))
        else:
            run_llm(cfg, asr, run_dir / "llm.jsonl", "hypothesis", task)
            merged = asr.merge(read_jsonl(run_dir / "llm.jsonl"), on="item_id")
    merged["is_warmup"] = _warmup_flags(merged, cfg["latency"]["warmup_items"]).values
    merged["e2e_s"] = merged.asr_s + merged.llm_s
    merged.to_json(run_dir / "predictions.jsonl", orient="records", lines=True, force_ascii=False)
    return merged
