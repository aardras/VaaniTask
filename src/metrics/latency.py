"""Latency summaries. Warm-up items are excluded by the caller via `is_warmup`."""
import numpy as np
import pandas as pd


def add_rtf(df: pd.DataFrame) -> pd.DataFrame:
    """RTF = processing time / audio duration (< 1 means faster than real time)."""
    return df.assign(asr_rtf=df.asr_s / df.duration_s, e2e_rtf=df.e2e_s / df.duration_s,
                     tok_per_s=df.out_tokens / (df.llm_s - df.ttft_s).clip(lower=1e-6))


def summarize(df: pd.DataFrame) -> dict:
    d = add_rtf(df[~df.is_warmup])
    out = {"n": len(d)}
    for col in ["asr_s", "ttft_s", "llm_s", "e2e_s", "asr_rtf", "e2e_rtf", "tok_per_s"]:
        x = d[col].to_numpy(dtype=float)
        x = x[~np.isnan(x)]  # LLM columns are NaN in --asr-only runs
        if len(x) == 0:
            continue
        out[f"{col}_mean"] = float(np.mean(x))
        out[f"{col}_p50"] = float(np.percentile(x, 50))
        out[f"{col}_p90"] = float(np.percentile(x, 90))
    return out
