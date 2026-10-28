"""FLEURS download, reproducible sampling, and noise augmentation.

Raw FLEURS files are fetched straight from the HF repo (TSV + audio tarball) so
we don't depend on the dataset loading script, which datasets>=3 no longer runs.
English gold = en_us raw_transcription joined on the shared FLoRes sentence id.
"""
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download

REPO = "google/fleurs"
SR = 16000
TSV_COLS = ["id", "file_name", "raw_transcription", "transcription",
            "phonemes", "num_samples", "gender"]


def _download(lang: str, split: str, root: Path, audio: bool) -> tuple[pd.DataFrame, Path]:
    tsv = hf_hub_download(REPO, f"data/{lang}/{split}.tsv", repo_type="dataset")
    df = pd.read_csv(tsv, sep="\t", header=None, names=TSV_COLS, quoting=3)
    audio_dir = root / "raw" / lang / split
    if audio and not audio_dir.exists():
        tar = hf_hub_download(REPO, f"data/{lang}/audio/{split}.tar.gz", repo_type="dataset")
        audio_dir.mkdir(parents=True)
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.isfile() and m.name.endswith(".wav"):
                    m.name = Path(m.name).name  # flatten, and refuse any path components
                    tf.extract(m, audio_dir)
    return df, audio_dir


def english_gold(root: Path, split: str = "test") -> dict[int, str]:
    en, _ = _download("en_us", split, root, audio=False)
    return en.drop_duplicates("id").set_index("id")["raw_transcription"].to_dict()


def build_manifest(lang: str, n: int, seed: int, root: Path, split: str = "test") -> pd.DataFrame:
    """Sample n utterances (one per FLoRes sentence, gender-stratified) that have English gold."""
    df, audio_dir = _download(lang, split, root, audio=True)
    gold = english_gold(root, split)
    df = df[df["id"].isin(gold)].copy()
    df = df[df["file_name"].map(lambda f: (audio_dir / f).exists())]
    rng = np.random.default_rng(seed)
    # One recording per sentence so the sample covers n distinct sentences.
    df = df.sample(frac=1, random_state=rng.integers(1 << 31)).drop_duplicates("id")
    per_gender = df.groupby("gender", group_keys=False)
    k = {g: round(n * len(d) / len(df)) for g, d in per_gender}
    k[max(k, key=k.get)] += n - sum(k.values())
    sample = pd.concat([d.head(k[g]) for g, d in per_gender])
    sample = sample.assign(
        lang=lang,
        audio_path=sample["file_name"].map(lambda f: str(audio_dir / f)),
        duration_s=sample["num_samples"] / SR,
        english_gold=sample["id"].map(gold),
    )
    # Remaining recordings (unused sentences) are the babble-noise pool.
    pool = df[~df["id"].isin(sample["id"])]["file_name"].map(lambda f: str(audio_dir / f))
    return sample.reset_index(drop=True), pool.tolist()


def load_audio(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
    return x


def make_noise(n_samples: int, babble_pool: list[str], rng: np.random.Generator,
               n_talkers: int = 4, white_frac: float = 0.3) -> np.ndarray:
    """Babble (sum of n_talkers random utterances) mixed with white noise, unit RMS."""
    babble = np.zeros(n_samples, dtype=np.float32)
    for p in rng.choice(babble_pool, size=n_talkers, replace=False):
        b = load_audio(p)
        b = np.tile(b, int(np.ceil(n_samples / len(b))))
        off = rng.integers(0, len(b) - n_samples + 1)
        b = b[off:off + n_samples]
        babble += b / (np.sqrt(np.mean(b ** 2)) + 1e-9)
    babble /= np.sqrt(np.mean(babble ** 2)) + 1e-9
    white = rng.standard_normal(n_samples).astype(np.float32)
    noise = np.sqrt(1 - white_frac) * babble + np.sqrt(white_frac) * white
    return noise / (np.sqrt(np.mean(noise ** 2)) + 1e-9)


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    p_sig = np.mean(clean ** 2)
    scaled = noise * np.sqrt(p_sig / (10 ** (snr_db / 10)))
    mixed = clean + scaled
    peak = np.abs(mixed).max()
    return mixed / peak * 0.99 if peak > 0.99 else mixed  # avoid clipping on write
