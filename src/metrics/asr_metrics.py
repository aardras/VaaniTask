"""WER/CER for Indic text with language-aware normalisation.

Whisper's BasicTextNormalizer is deliberately NOT used: it strips Unicode
combining marks (matras, virama, anusvara), which turns Indic text into
consonant skeletons and makes WER/CER meaningless.
"""
import re
import unicodedata

import jiwer
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory

_FACTORY = IndicNormalizerFactory()
_NORMALIZERS = {}

# ASCII + general punctuation + danda/double danda; keeps letters, marks, digits.
_PUNCT_RE = re.compile(r"[।॥ -⁯!-/:-@[-`{-~]")
_WS_RE = re.compile(r"\s+")


def _lang_code(lang: str) -> str:
    """'hi_in' -> 'hi'."""
    return lang.split("_")[0]


def _normalizer(lang: str):
    code = _lang_code(lang)
    if code not in _NORMALIZERS:
        # remove_nuktas=False: nukta changes the phoneme in Hindi (ज vs ज़).
        _NORMALIZERS[code] = _FACTORY.get_normalizer(code, remove_nuktas=False)
    return _NORMALIZERS[code]


def normalize(text: str, lang: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = _normalizer(lang).normalize(text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip().lower()  # lower() only affects Latin code-mixed tokens
    return text


def wer_cer(refs: list[str], hyps: list[str], lang: str) -> dict:
    """Corpus-level WER and CER (errors summed over all utterances / total ref length)."""
    r = [normalize(x, lang) for x in refs]
    h = [normalize(x, lang) for x in hyps]
    # jiwer rejects empty references; FLEURS refs are never empty, but guard anyway.
    pairs = [(a, b) for a, b in zip(r, h) if a]
    r, h = [p[0] for p in pairs], [p[1] for p in pairs]
    return {"wer": jiwer.wer(r, h), "cer": jiwer.cer(r, h), "n": len(r)}


def is_degenerate(hyp: str, ref: str, lang: str) -> bool:
    """Whisper failure mode: repetition loops / runaway output.

    Flagged when >50% of hypothesis words are repeats, a long hypothesis uses
    almost no distinct characters ("ḍḍḍḍ…"), or it is more than 2x the reference
    length in characters (the decoder ran to its token cap)."""
    h = normalize(hyp, lang)
    words = h.split()
    if not words:
        return False
    word_loop = len(words) >= 6 and 1 - len(set(words)) / len(words) > 0.5
    char_loop = len(h) >= 20 and len(set(h.replace(" ", ""))) <= 3
    return word_loop or char_loop or len(h) > 2 * len(normalize(ref, lang))


def utterance_errors(ref: str, hyp: str, lang: str) -> dict:
    """Per-utterance error counts so corpus WER/CER can be bootstrapped later."""
    r, h = normalize(ref, lang), normalize(hyp, lang)
    w = jiwer.process_words(r, h)
    c = jiwer.process_characters(r, h)
    return {
        "word_errors": w.substitutions + w.deletions + w.insertions,
        "ref_words": len(r.split()),
        "char_errors": c.substitutions + c.deletions + c.insertions,
        "ref_chars": len(r),
        "degenerate": is_degenerate(hyp, ref, lang),
    }
