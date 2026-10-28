"""Download FLEURS, sample the evaluation set, and render noisy copies.

Output: data/manifest.csv with one row per (utterance, noise condition).
    python scripts/prepare_data.py --config configs/baseline.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import SR, add_noise, build_manifest, load_audio, make_noise  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    dc, seed = cfg["data"], cfg["seed"]
    root = Path(dc["root"])

    rows = []
    for li, lang in enumerate(dc["languages"]):
        sample, pool = build_manifest(lang, dc["n_per_language"], seed, root)
        print(f"{lang}: {len(sample)} utterances, {sample.duration_s.sum() / 60:.1f} min, "
              f"gender={sample.gender.value_counts().to_dict()}, babble pool={len(pool)}")
        for i, r in tqdm(sample.iterrows(), total=len(sample), desc=lang):
            clean = load_audio(r.audio_path)
            # Same noise realisation for every SNR of an item: only the level varies.
            noise = make_noise(len(clean), pool, np.random.default_rng([seed, li, i]))
            for snr in dc["snr_db"]:
                if snr == "clean":
                    path = r.audio_path
                else:
                    out = root / "audio" / lang / f"snr{snr}" / f"{r.id}.wav"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(out, add_noise(clean, noise, float(snr)), SR)
                    path = str(out)
                rows.append({
                    "item_id": f"{lang}_{r.id}_{snr}", "utt_id": f"{lang}_{r.id}",
                    "lang": lang, "snr": str(snr), "gender": r.gender,
                    "audio_path": path, "duration_s": r.duration_s,
                    "reference": r.raw_transcription, "english_gold": r.english_gold,
                })
    out = root / "manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
