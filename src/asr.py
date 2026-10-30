"""ASR backends behind one interface: transcribe(audio, lang) -> (text, seconds).

Both backends use greedy decoding with the language forced, so baseline vs
improved differ only in model weights / inference engine.
"""
import time
from pathlib import Path

import numpy as np
import torch


def split_spec(spec: str) -> tuple[str, str | None]:
    """'repo@commit' -> ('repo', 'commit'); plain 'repo' -> ('repo', None).

    Pinning a revision lets us load a repo's safetensors conversion PR: transformers
    refuses pickle .bin weights under torch < 2.6 (CVE-2025-32434), and we keep torch
    2.5.1 so all runs are timed on the same stack."""
    repo, _, rev = spec.partition("@")
    return repo, rev or None


_BASE_BY_WIDTH = {384: "tiny", 512: "base", 768: "small", 1024: "medium", 1280: "large-v2"}


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class HFWhisperASR:
    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "float16"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.dtype = getattr(torch, dtype)
        self.device = device
        repo, rev = split_spec(model_id)
        self.processor = WhisperProcessor.from_pretrained(repo, revision=rev)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            repo, revision=rev, dtype=self.dtype, use_safetensors=True).to(device).eval()
        if not hasattr(self.model.generation_config, "lang_to_id"):
            # Older fine-tunes (e.g. vasista22/whisper-hindi-small) ship a generation config
            # without language tokens, so generate(language=...) refuses it. Use the base
            # model's config, as recommended in transformers issue #25084.
            from transformers import GenerationConfig
            size = _BASE_BY_WIDTH[self.model.config.d_model]
            self.model.generation_config = GenerationConfig.from_pretrained(f"openai/whisper-{size}")

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
        repo, rev = split_spec(model_id)
        path = Path(cache) / (repo.replace("/", "__") + (f"@{rev[:10]}" if rev else ""))
        if not (path / "model.bin").exists():
            from ctranslate2.converters import TransformersConverter
            from huggingface_hub import snapshot_download
            from transformers import WhisperProcessor
            # The converter doesn't request safetensors, so give it a snapshot with no pickle
            # .bin files at all (torch < 2.6 must not torch.load them; see split_spec).
            local = snapshot_download(repo, revision=rev, ignore_patterns=["*.bin", "*.msgpack", "*.h5"])
            TransformersConverter(local).convert(str(path), quantization=dtype, force=True)
            # Not every fine-tuned repo ships tokenizer.json (vasista22/whisper-hindi-small
            # doesn't); write it and the feature-extractor config from the HF processor so
            # faster-whisper uses this checkpoint's own tokenizer and mel settings.
            WhisperProcessor.from_pretrained(repo, revision=rev).save_pretrained(str(path))
        self.model = WhisperModel(str(path), device=device, compute_type=dtype)
        self._prompts = {}

    # CTranslate2 4.8.4's Whisper generate() returns max_length/2 new tokens (measured:
    # 100->50, 200->100, 444->222), so faster-whisper's transcribe() caps output at ~222
    # tokens -- about 100 Malayalam characters, since Whisper's byte-level BPE spends ~2.2
    # tokens per Malayalam character. That halved every long Malayalam transcript (CER
    # 26.6% vs 5.2% for the same weights on HF). We call CTranslate2 directly with the
    # limit doubled; on 8 Malayalam items this matched HF output (5/8 identical, equal CER).
    MAX_NEW_TOKENS = 440  # same budget as HFWhisperASR

    def _prompt(self, lang: str) -> list[int]:
        if lang not in self._prompts:
            from faster_whisper.tokenizer import Tokenizer
            tok = Tokenizer(self.model.hf_tokenizer, True, task="transcribe", language=lang.split("_")[0])
            self._prompts[lang] = (tok, list(tok.sot_sequence) + [tok.no_timestamps])
        return self._prompts[lang]

    def transcribe(self, audio: np.ndarray, lang: str) -> tuple[str, float]:
        tok, prompt = self._prompt(lang)
        _sync()
        t0 = time.perf_counter()
        # Same input handling as HFWhisperASR: one 30 s window (longer audio is truncated,
        # as WhisperProcessor does), greedy, <|notimestamps|>.
        feats = self.model.feature_extractor(audio)[:, :3000]
        feats = np.pad(feats, ((0, 0), (0, 3000 - feats.shape[1])))
        res = self.model.model.generate(self.model.encode(feats), [prompt], beam_size=1,
                                        max_length=2 * self.MAX_NEW_TOKENS)[0]
        text = tok.decode(res.sequences_ids[0]).strip()
        _sync()
        return text, time.perf_counter() - t0


BACKENDS = {"hf": HFWhisperASR, "faster_whisper": FasterWhisperASR}


def load_asr(backend: str, model_id: str, device: str, dtype: str):
    return BACKENDS[backend](model_id, device=device, dtype=dtype)
