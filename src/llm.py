"""Instruction-tuned LLM (4-bit, greedy) that restates an Indic transcript in the same language."""
import time

import torch

LANG_NAMES = {"hi_in": "Hindi", "ml_in": "Malayalam"}
SCRIPTS = {"hi_in": "Devanagari", "ml_in": "Malayalam"}

SYSTEM_PROMPT = (
    "You are a voice assistant for Indian users. You receive an automatic speech "
    "recognition transcript of what the user said in {language}; it may contain "
    "recognition errors, misspellings, repeated words, or words in the wrong script. "
    "Respond with a faithful, fluent restatement of what the user said, in {language}, "
    "written in {script} script. Fix words that look misrecognised using the most "
    "plausible meaning from context, but never add facts, names or numbers that were "
    "not spoken, and do not translate into English. Output only the {language} sentence."
)


def build_messages(transcript: str, lang: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(language=LANG_NAMES[lang],
                                                           script=SCRIPTS[lang])},
        {"role": "user", "content": transcript},
    ]


class _FirstTokenTimer:
    """Streamer that only timestamps the first generated token.

    generate() calls put() once with the prompt, then once per new token, so the
    2nd call is the first token. Token-level (unlike TextIteratorStreamer, which
    only emits at word boundaries and so misreports TTFT for space-free output)."""

    def __init__(self):
        self.calls, self.first = 0, None

    def put(self, value):
        self.calls += 1
        if self.calls == 2:
            torch.cuda.synchronize()
            self.first = time.perf_counter()

    def end(self):
        pass


class LLM:
    def __init__(self, model_id: str, load_in_4bit: bool = True, max_new_tokens: int = 128,
                 quantize_lm_head: bool = False):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        # bitsandbytes leaves lm_head in fp16 by default. For a 152k-vocab 7B model that is
        # ~1.1 GB, which pushes the judge past 6 GB VRAM into slow shared memory on Windows;
        # an empty skip list quantizes it too.
        skip = [] if quantize_lm_head else None
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.float16,
                                   llm_int8_skip_modules=skip) if load_in_4bit else None
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=quant, dtype=torch.float16, device_map="cuda").eval()
        self.max_new_tokens = max_new_tokens

    def chat(self, messages: list[dict], max_new_tokens: int | None = None) -> dict:
        """Greedy generation; returns text plus TTFT, total time and output token count."""
        enc = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt").to(self.model.device)
        timer = _FirstTokenTimer()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(**enc, streamer=timer,
                                  max_new_tokens=max_new_tokens or self.max_new_tokens,
                                  do_sample=False, temperature=None, top_p=None, top_k=None,
                                  pad_token_id=self.tok.eos_token_id)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
        new = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        return {"text": text, "ttft_s": (timer.first - t0) if timer.first else total,
                "llm_s": total, "out_tokens": int((new != self.tok.eos_token_id).sum())}

    TOKENS_PER_SPEECH_SECOND = 30  # max over our FLEURS refs is 22.3 (Malayalam), +35% headroom

    def respond(self, transcript: str, lang: str, duration_s: float) -> dict:
        # A restatement can't say more than was spoken, so the token cap scales with audio
        # duration. Indic text is token-heavy (Malayalam refs average ~200 Qwen3 tokens), so
        # one fixed budget would truncate long answers; capping by transcript length instead
        # would hand looped ASR output (hundreds of tokens) a huge budget to copy.
        cap = int(min(self.max_new_tokens, self.TOKENS_PER_SPEECH_SECOND * duration_s + 32))
        return self.chat(build_messages(transcript, lang), max_new_tokens=cap)
