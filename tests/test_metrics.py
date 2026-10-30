"""Known-answer tests for our own evaluation code.  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics.asr_metrics import is_degenerate, normalize, utterance_errors, wer_cer  # noqa: E402
from src.metrics.response_metrics import (_numbers, cohen_kappa,  # noqa: E402
                                          heuristic_hallucination, off_script, parse_judge)
from src.stats import paired_diff_ci, ratio_ci  # noqa: E402


def test_identical_is_zero():
    s = "भारत एक विशाल देश है।"
    assert wer_cer([s], [s], "hi_in") == {"wer": 0.0, "cer": 0.0, "n": 1}


def test_punctuation_and_danda_ignored():
    assert normalize("नमस्ते, दुनिया।", "hi_in") == normalize("नमस्ते दुनिया", "hi_in")


def test_matras_are_kept():
    # BasicTextNormalizer would map both to the same skeleton; we must not.
    assert normalize("कमल", "hi_in") != normalize("कोमल", "hi_in")


def test_one_substitution_in_four_words():
    r = utterance_errors("मैं घर जा रहा", "मैं घर आ रहा", "hi_in")
    assert r["word_errors"] == 1 and r["ref_words"] == 4


def test_malayalam_chillu_forms_equal():
    # Atomic chillu ൻ (U+0D7B) vs legacy न + virama + ZWJ sequence must compare equal.
    atomic = "അവൻ"
    legacy = "അവന്‍"
    assert normalize(atomic, "ml_in") == normalize(legacy, "ml_in")


def test_normalize_idempotent():
    for s, l in [("भारत, एक देश।", "hi_in"), ("കേരളം ഒരു സംസ്ഥാനമാണ്.", "ml_in")]:
        once = normalize(s, l)
        assert normalize(once, l) == once


def test_degenerate_loops_detected():
    ref = "ചെറിയ ദ്വീപുകളിൽ ഭൂരിഭാഗവും സ്വതന്ത്ര രാജ്യങ്ങളാണ്"
    assert is_degenerate("ḍ" * 140, ref, "ml_in")                       # char loop
    assert is_degenerate(" ".join(["അഗർ"] * 30), ref, "ml_in")         # word loop
    assert is_degenerate(ref + " " + ref + " " + ref, ref, "ml_in")      # runaway length
    assert not is_degenerate(ref, ref, "ml_in")
    assert not is_degenerate("", ref, "ml_in")


def test_numbers():
    assert _numbers("10 people and 1,000 dollars, ten cats") == {"10", "1000"}
    assert _numbers("a million") == _numbers("1000000")


def test_indic_digits_map_to_ascii():
    assert _numbers("२०१२ में") == {"2012"} and _numbers("൨൦൧൨ ൽ") == {"2012"}


def test_heuristic_new_number():
    ref, gold = "पुल 2012 में खुला", "The bridge opened in 2012."
    assert not heuristic_hallucination("पुल २०१२ में खुला", ref, gold)["heur_halluc"]
    r = heuristic_hallucination("पुल 2015 में खुला", ref, gold)
    assert r["heur_new_numbers"] == ["2015"] and r["heur_halluc"]


def test_heuristic_number_supported_by_english_gold():
    # Native transcript spells the number out; English gold has digits -> supported.
    ref, gold = "पुल दो हज़ार बारह में खुला", "The bridge opened in 2012."
    assert not heuristic_hallucination("पुल 2012 में खुला", ref, gold)["heur_halluc"]


def test_heuristic_runaway_is_not_hallucination():
    ref = "पुल खुला"
    r = heuristic_hallucination("पुल खुला " * 5, ref, "The bridge opened.")
    assert r["heur_runaway"] and not r["heur_halluc"]


def test_off_script():
    assert not off_script("ചെറിയ ദ്വീപുകളിൽ", "ml_in")
    assert off_script("In this video, I will show you", "ml_in")
    assert off_script("अगर अगर अगर", "ml_in")       # Hindi script in a Malayalam response
    assert not off_script("भारत 2012 में", "hi_in")  # digits don't count against script


def test_parse_judge():
    assert parse_judge('ok {"label": "partial", "hallucination": true, "unsupported_span": "x"}') == \
        {"judge_label": "partial", "judge_halluc": True, "judge_span": "x", "judge_parse_ok": True}
    assert parse_judge("garbage")["judge_parse_ok"] is False


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_ratio_ci_contains_point():
    num, den = np.array([1, 2, 0, 3]), np.array([5, 5, 5, 5])
    p, lo, hi = ratio_ci(num, den)
    assert p == pytest.approx(0.3) and lo <= p <= hi


def test_paired_diff_zero_when_equal():
    a = np.array([1, 2, 3.0])
    assert paired_diff_ci(a, a)["diff"] == 0


def test_parse_reply_judge():
    from src.metrics.response_metrics import parse_reply_judge
    r = parse_reply_judge('Here: {"relevance": "on_topic", "misattribution": false, '
                          '"factual_hallucination": "true", "clarification": false, "span": "x"}')
    assert r["rj_relevance"] == "on_topic" and r["rj_factual_halluc"] is True and r["rj_parse_ok"]
    assert parse_reply_judge('{"relevance": "off-topic", "misattribution": true}')["rj_relevance"] == "off_topic"
    bad = parse_reply_judge("I think it is relevant")
    assert bad["rj_parse_ok"] is False and bad["rj_relevance"] == "off_topic"


def test_parse_reply_judge_malformed_json_fallback():
    from src.metrics.response_metrics import parse_reply_judge
    raw = '{"claim": "none", "relevance": "partial", "misattribution": true, "factual_hallucination": false, ' \
          '"clarification": false, "span": "he said "this" here"}'  # unescaped quotes -> invalid JSON
    r = parse_reply_judge(raw)
    assert r["rj_parse_ok"] and r["rj_relevance"] == "partial" and r["rj_misattribution"] is True
