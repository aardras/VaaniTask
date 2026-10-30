"""Spoken language identification, restricted to the languages the pipeline supports.

Stock multilingual Whisper predicts the language token right after <|startoftranscript|>.
We run one decoder step and compare only the candidate language tokens, so detection
can't drift to a language we have no ASR model for. Fine-tuned checkpoints are not used
here: single-language fine-tunes no longer predict language tokens reliably.
"""
import time

import numpy as np
import torch

CANDIDATES = {"hi": "hi_in", "ml": "ml_in"}


class LanguageID:
    # whisper-medium over small: Malayalam 0 dB LID 86.7% vs 70.0%, for ~65 ms more per utterance.
    def __init__(self, model_id: str = "openai/whisper-medium", device: str = "cuda",
                 dtype: str = "float16"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.device, self.dtype = device, getattr(torch, dtype)
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype, use_safetensors=True).to(device).eval()
        tok = self.processor.tokenizer
        self.sot = tok.convert_tokens_to_ids("<|startoftranscript|>")
        self.lang_ids = {code: tok.convert_tokens_to_ids(f"<|{code}|>") for code in CANDIDATES}

    @torch.inference_mode()
    def detect(self, audio: np.ndarray) -> tuple[str, dict[str, float], float]:
        """Returns (lang like 'hi_in', {lang: probability}, seconds)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        logits = self.model(input_features=feats.to(self.device, self.dtype),
                            decoder_input_ids=torch.tensor([[self.sot]], device=self.device)).logits[0, -1]
        codes = list(self.lang_ids)
        probs = torch.softmax(logits[[self.lang_ids[c] for c in codes]].float(), dim=-1).tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        dist = {CANDIDATES[c]: p for c, p in zip(codes, probs)}
        return max(dist, key=dist.get), dist, time.perf_counter() - t0
