"""Bootstrap confidence intervals (utterance-level resampling)."""
import numpy as np

N_BOOT = 2000


def ratio_ci(num: np.ndarray, den: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    """Corpus-level ratio sum(num)/sum(den) (e.g. WER = errors / ref words) with 95% CI."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(N_BOOT, len(num)))
    boots = num[idx].sum(1) / den[idx].sum(1)
    return num.sum() / den.sum(), *np.percentile(boots, [2.5, 97.5])


def mean_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    return ratio_ci(x, np.ones_like(x), seed)


def paired_diff_ci(a: np.ndarray, b: np.ndarray, den: np.ndarray | None = None,
                   seed: int = 0) -> dict:
    """Difference b - a on the same items (improved - baseline), 95% CI and two-sided p-value.

    With `den`, a and b are per-item numerators of a corpus ratio (e.g. word errors)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    den = np.ones_like(a) if den is None else np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    d = (b[idx].sum(1) - a[idx].sum(1)) / den[idx].sum(1)
    point = (b.sum() - a.sum()) / den.sum()
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return {"diff": point, "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
            "p": min(p, 1.0)}
