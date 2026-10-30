"""Speech in, text response out, for one recording.

    python scripts/demo.py my_recording.wav                 # auto-detects Hindi vs Malayalam
    python scripts/demo.py my_recording.wav --lang ml_in    # skip detection
    python scripts/demo.py my_recording.wav --task both     # reply + restatement

Any format/sample rate librosa can read; audio is resampled to 16 kHz mono and only the
first 30 s are transcribed (one Whisper window). Models load one at a time (6 GB GPU).
"""
import argparse
import gc
import sys
from pathlib import Path

import librosa
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.asr import load_asr  # noqa: E402
from src.lid import LanguageID  # noqa: E402
from src.llm import LANG_NAMES, LLM  # noqa: E402


def _free():
    # Callers must `del` their own reference first: deleting a parameter here would only drop
    # the local name, leaving the model on the GPU next to the LLM (-> shared-memory spill, ~10x slower).
    gc.collect()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--lang", choices=["auto", "hi_in", "ml_in"], default="auto")
    ap.add_argument("--task", choices=["reply", "restate", "both"], default="reply")
    ap.add_argument("--config", default="configs/improved.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    dur = len(audio) / 16000
    print(f"audio: {args.audio} ({dur:.1f} s)" + ("  [only the first 30 s are transcribed]" if dur > 30 else ""))

    lang = args.lang
    if lang == "auto":
        lid = LanguageID()
        lang, dist, secs = lid.detect(audio)
        del lid
        _free()
        print(f"language: {LANG_NAMES[lang]} (p={dist[lang]:.2f}, {secs:.2f} s)")
    else:
        print(f"language: {LANG_NAMES[lang]} (given)")

    ac = cfg["asr"]
    asr = load_asr(ac["backend"], ac["models"][lang], ac["device"], ac["dtype"])
    transcript, asr_s = asr.transcribe(audio, lang)
    del asr
    _free()
    print(f"\ntranscript ({asr_s:.2f} s, RTF {asr_s / dur:.3f}):\n  {transcript}")

    lc = cfg["llm"]
    llm = LLM(lc["model"], lc["load_in_4bit"], lc["max_new_tokens"])
    llm.chat([{"role": "user", "content": "नमस्ते"}], max_new_tokens=4)  # warm-up so timings are representative
    for task in (["reply", "restate"] if args.task == "both" else [args.task]):
        res = llm.respond(transcript, lang, dur, task)
        print(f"\n{task} (first token {res['ttft_s']:.2f} s, total {res['llm_s']:.2f} s):\n  {res['text']}")
    del llm
    _free()


if __name__ == "__main__":
    main()
