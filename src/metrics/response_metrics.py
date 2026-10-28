"""Response correctness and hallucination for in-language restatements.

The LLM's task is to restate what the user said, in the same language and script,
fixing ASR errors. Each response is scored against two references:
  native reference - the FLEURS transcript (surface form: chrF++, response CER)
  English gold     - the parallel FLoRes sentence (meaning: cross-lingual LaBSE, judge)

Definitions (also stated in REPORT.md):
  correct    - in the target language and conveys every key element of the
               reference (actors, actions, entities, numbers, places, times);
               paraphrase is fine.
  partial    - main idea preserved but >=1 key element missing or wrong.
  incorrect  - main idea missing/wrong, unrelated, empty, or not in the target language.
  hallucination - the response asserts a specific detail (entity, number, event,
               attribute) that is NOT stated or implied by the reference. Omissions
               and wrong-but-substituted words count against correctness, not here,
               unless the substitute introduces new specific content.
Two independent hallucination detectors: an LLM judge (primary) and a lexical
heuristic (new numbers only; high precision, low recall; cross-check).
"""
import json
import re

import numpy as np
import sacrebleu

# Unicode blocks of the target scripts.
SCRIPT_RANGES = {"hi_in": (0x0900, 0x097F), "ml_in": (0x0D00, 0x0D7F)}
LANG_NAMES = {"hi_in": "Hindi", "ml_in": "Malayalam"}

# ---------- automatic similarity ----------

def chrf(hyps: list[str], refs: list[str]) -> list[float]:
    return [sacrebleu.sentence_chrf(h or "", [r], word_order=2).score for h, r in zip(hyps, refs)]


class LaBSE:
    """Language-agnostic sentence embeddings: comparable across Hindi/Malayalam/English."""

    def __init__(self, device: str = "cuda"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    def similarity(self, hyps: list[str], refs: list[str]) -> list[float]:
        a = self.model.encode([h or "" for h in hyps], batch_size=64, normalize_embeddings=True)
        b = self.model.encode(refs, batch_size=64, normalize_embeddings=True)
        return (a * b).sum(axis=1).tolist()


def target_script_fraction(text: str, lang: str) -> float:
    """Share of letters in the target script (Latin output = translated / transliterated)."""
    lo, hi = SCRIPT_RANGES[lang]
    letters = [c for c in text if c.isalpha() or lo <= ord(c) <= hi]
    if not letters:
        return 0.0
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def off_script(text: str, lang: str) -> bool:
    return target_script_fraction(text, lang) < 0.8


# ---------- lexical hallucination heuristic ----------

# Devanagari ०-९ and Malayalam ൦-൯ -> ASCII
_DIGITS = str.maketrans({**{chr(0x0966 + i): str(i) for i in range(10)},
                         **{chr(0x0D66 + i): str(i) for i in range(10)}})
_NUM_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
              "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
              "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
              "thousand": 1000, "million": 10 ** 6, "billion": 10 ** 9}


def _numbers(text: str) -> set[str]:
    text = text.translate(_DIGITS)
    # "1,000" / "1000.0" / "1000" all -> "1000"
    digits = {f"{float(d.replace(',', '')):g}" for d in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}
    words = {f"{float(_NUM_WORDS[w]):g}" for w in re.findall(r"[a-z]+", text.lower()) if w in _NUM_WORDS}
    return digits | words


def heuristic_hallucination(response: str, reference: str, english_gold: str) -> dict:
    """A number is supported if it appears in either reference (the native transcript may
    spell it out in words where the English gold uses digits).

    Runaway length (>1.5x the reference) is reported separately and NOT counted as
    hallucination: in practice it is the LLM copying an ASR repetition loop, which is a
    correctness failure rather than invented content."""
    new_nums = sorted(_numbers(response) - _numbers(reference) - _numbers(english_gold))
    runaway = len(response.strip()) > 1.5 * len(reference.strip())
    return {"heur_halluc": bool(new_nums), "heur_new_numbers": new_nums, "heur_runaway": runaway}


# ---------- LLM judge ----------

JUDGE_PROMPT = """You are grading a speech assistant. The user spoke one sentence in {language}. The assistant had to restate what the user said, in {language}, fixing speech-recognition errors without adding anything.

REFERENCE ({language}, exactly what the user said): {reference}
MEANING (English translation of the reference): {gold}
RESPONSE (the assistant's restatement): {response}

Grade with these rules:
- "correct": the response is in {language} and conveys every key element of the reference (who, what, entities, numbers, places, times). Paraphrasing and different spellings are fine.
- "partial": the main idea is preserved but at least one key element is missing or wrong.
- "incorrect": the main idea is missing or wrong, the response is unrelated, empty, or not in {language}.
- "hallucination": true if the response states any specific detail (a name, number, event, place or attribute) that is NOT stated or implied by the reference. Merely omitting information, or garbling a word, is NOT hallucination.

Reply with JSON only: {{"label": "correct|partial|incorrect", "hallucination": true|false, "unsupported_span": "<the invented detail, or empty>"}}"""

LABEL_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    label = str(d.get("label", "")).lower().strip()
    if label not in LABEL_SCORE:  # unparseable -> worst case, but flagged
        return {"judge_label": "incorrect", "judge_halluc": False, "judge_span": "", "judge_parse_ok": False}
    h = d.get("hallucination", False)
    h = h if isinstance(h, bool) else str(h).lower() == "true"
    return {"judge_label": label, "judge_halluc": h,
            "judge_span": str(d.get("unsupported_span", "")), "judge_parse_ok": True}


def judge_one(llm, response: str, reference: str, gold: str, lang: str) -> dict:
    """`llm` is a src.llm.LLM loaded with the judge model."""
    if not (response or "").strip():
        return {"judge_label": "incorrect", "judge_halluc": False, "judge_span": "",
                "judge_parse_ok": True}
    msg = [{"role": "user", "content": JUDGE_PROMPT.format(
        language=LANG_NAMES[lang], reference=reference, gold=gold, response=response)}]
    return parse_judge(llm.chat(msg, max_new_tokens=160)["text"])


def cohen_kappa(a: list, b: list) -> float:
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in cats)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0
