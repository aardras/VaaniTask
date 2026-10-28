"""ASR backends behind one interface: transcribe(audio, lang) -> (text, seconds).

Both backends use greedy decoding with the language forced, so baseline vs
improved differ only in model weights / inference engine.
"""
import time
from pathlib import Path

import numpy as np
import torch


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class HFWhisperASR:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.dtype = getattr(torch, dtype)
        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            model_id, dtype=self.dtype).to(device).eval()

    @torch.inference_mode()
    def transcribe(self, audio: np.ndarray, lang: str) -> tuple[str, float]:
        _sync()
        t0 = time.perf_counter()
        feats = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        ids = self.model.generate(feats.to(self.device, self.dtype),
                                  language=lang.split("_")[0], task="transcribe",
                                  num_beams=1, do_sample=False, max_new_tokens=440)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        _sync()
        return text, time.perf_counter() - t0


class FasterWhisperASR:
    """CTranslate2 engine. HF checkpoints are converted once and cached under models/ct2/."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16",
                 cache: str = "models/ct2"):
        from faster_whisper import WhisperModel
        path = Path(cache) / model_id.replace("/", "__")
        if not (path / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter
            from transformers import WhisperProcessor
            TransformersConverter(model_id).convert(str(path), quantization=dtype, force=True)
            # Not every fine-tuned repo ships tokenizer.json (vasista22/whisper-hindi-small
            # doesn't); write it and the feature-extractor config from the HF processor so
            # faster-whisper uses this checkpoint's own tokenizer and mel settings.
            WhisperProcessor.from_pretrained(model_id).save_pretrained(str(path))
        self.model = WhisperModel(str(path), device=device, compute_type=dtype)

    def transcribe(self, audio: np.ndarray, lang: str) -> tuple[str, float]:
        _sync()
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(audio, language=lang.split("_")[0], task="transcribe",
                                            beam_size=1, temperature=0.0, vad_filter=False,
                                            condition_on_previous_text=False)
        text = " ".join(s.text.strip() for s in segments).strip()  # generator: decoding happens here
        _sync()
        return text, time.perf_counter() - t0


BACKENDS = {"hf": HFWhisperASR, "faster_whisper": FasterWhisperASR}


def load_asr(backend: str, model_id: str, device: str, dtype: str):
    return BACKENDS[backend](model_id, device=device, dtype=dtype)
